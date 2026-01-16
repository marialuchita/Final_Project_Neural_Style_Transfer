import csv
import time
import os
from loss_network import *
from transform_network import TransformNetwork



import torch
import numpy as np
from data_pipeline import *
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

BATCH_SIZE = 4
WORKERS = 2
LEARNING_RATE = 1e-3
EPOCHS = 100

CONTENT_LAYER = "relu2_2"
STYLE_LAYERS = "relu2_2"
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 4e5 # Johnson uses 1e5 to 4e5
TV_WEIGHT = 1e-6 # 0 or 1e-6 to 1e-4



def process_style_img(style_img_path: str, device: torch.device) -> torch.Tensor:
    style_img = Image.open(style_img_path).convert("RGB")
    transform = transforms.ToTensor() # transforms to 0 - 1

    # Add one more dimension at index 0
    image_tensor = transform(style_img).unsqueeze(0).to(device) # (1, C, H, W)

    # Create a batch of BATCH-SIZE images, where each image in the batch is the style image. This is done so every content image has a matching style target.
    image_tensor = image_tensor.repeat(BATCH_SIZE, 1, 1, 1)  # (B, C, H, W)
    return image_tensor

def compute_tv_loss(t: torch.Tensor) -> torch.Tensor:
    # compute the differences between neighboring pixels
    horizontal_diff = torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])
    vertical_diff = torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])
    tv_loss = horizontal_diff.mean() + vertical_diff.mean()
    return tv_loss

def save_model(transformer: TransformNetwork, optimizer: Adam, folder_path: str, iteration: int) -> None:

    out_model_path = os.path.join(folder_path, f"model_at_iteration_{iteration}.pth")
    torch.save(
    {
            "transformer_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "content_weight": CONTENT_WEIGHT,
                "style_weight": STYLE_WEIGHT,
                "tv_weight": TV_WEIGHT
            }
        },
        out_model_path
    )
def get_log(epoch: int, time_elapsed: float, losses_dict: dict[str, float], iteration: int) -> dict[str, float]:
    log = {
        "epoch": epoch,
        "iteration": iteration,
        "content_loss": losses_dict["content_loss"] / iteration,
        "style_loss":  losses_dict["style_loss"] / iteration,
        "tv_loss": losses_dict["tv_loss"] / iteration,
        "total_loss": losses_dict["total_loss"] / iteration,
        "time_minutes": time_elapsed,
    }
    print(log)
    return log

def train(content_folder_path: str, style_img_name: str):
    style_img_path = os.path.join("images/style", style_img_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TrainDataset(content_folder_path)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    transformer_network = TransformNetwork().train().to(device)

    # Create and freeze the loss network (wrapper to vgg)
    vgg_network = LossNetwork().to(device)
    vgg_network.eval()
    vgg_network.requires_grad_(False)

    # Create Adam optimizer and pass in the parameters of the transformer network
    optimizer = torch.optim.Adam(transformer_network.parameters(), lr=LEARNING_RATE)

    # Load the style image, convert to tensor of shape (BATCH_SIZE, C, H, W)
    target_style_tensor = process_style_img(style_img_path, device)

    with torch.no_grad(): # no gradient should be computed. Frozen network
        target_style_features = vgg_network(normalize_for_vgg(target_style_tensor))
        target_style_gram_matrices = {k: gram_matrix(v) for k, v in target_style_features.items()}

    logs = []
    iteration = 0
    start_time = time.time()
    curr_time_elapsed = 0

    style_img_name_stem = Path(style_img_name).stem
    model_folder_path= os.path.join("models", style_img_name_stem)
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_name = os.path.join("logs", f"{style_img_name_stem}.csv")

    sum_losses = {"content_loss": 0, "style_loss": 0, "tv_loss": 0, "total_loss": 0}

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        for batch_index, content_batch in enumerate(loader):

            # Move content batch to device (cuda) if available >>>
            content_batch = content_batch.to(device)

            # -------------------------------------------------------------------------
            # Pass through transformer_network for stylization >>>
            # -------------------------------------------------------------------------
            stylized_batch = transformer_network(content_batch).clamp(0.0, 1.0)

            # -------------------------------------------------------------------------
            # Pass through loss_network to create feature maps >>>
            # -------------------------------------------------------------------------
            # Content feature maps:
            content_features = vgg_network(normalize_for_vgg(content_batch))
            # Style features maps and gram matrices:
            style_features = vgg_network(normalize_for_vgg(stylized_batch))
            style_gram_matrices = {k: gram_matrix(v) for k, v in style_features.items()}

            # -------------------------------------------------------------------------
            # Content loss >>>
            # -------------------------------------------------------------------------
            # Mean squared error of the feature maps of the content img
            # and feature maps of the stylized/passed through transformer network img.
            content_loss = CONTENT_WEIGHT * F.mse_loss(content_features[CONTENT_LAYER], style_features[CONTENT_LAYER])
            sum_losses["content_loss"] += content_loss.item()

            # -------------------------------------------------------------------------
            # Style loss >>>
            # -------------------------------------------------------------------------
            # MSE of the gram matrices of the feature maps of the style image
            # and the feature maps of the generated/stylized image
            style_loss = 0
            for layer, gram_m in style_gram_matrices.items():
                current_mse = F.mse_loss(gram_m, target_style_gram_matrices[layer])
                style_loss += current_mse

            style_loss = STYLE_WEIGHT * (style_loss / len(style_gram_matrices))
            sum_losses["style_loss"] += style_loss.item()

            if len(style_gram_matrices) != 4:
                print("ERROR style_loss should be divided by 4")

            # -------------------------------------------------------------------------
            # Total variation loss
            # -------------------------------------------------------------------------
            tv_loss = TV_WEIGHT * compute_tv_loss(stylized_batch)
            sum_losses["tv_loss"] += tv_loss.item()

            # -------------------------------------------------------------------------
            # Total loss
            # -------------------------------------------------------------------------
            total_loss = content_loss + style_loss + tv_loss
            sum_losses["total_loss"] += total_loss.item()

            optimizer.zero_grad() # clear the old gradients
            total_loss.backward() # compute new gradients
            optimizer.step() # update new weights

            iteration += 1
            curr_time_elapsed = (time.time() - start_time) / 60 # in minutes

            if batch_index % 500 == 0 or (epoch == EPOCHS - 1 and batch_index == len(loader) - 1 ) :
                log = get_log(
                    epoch=epoch + 1,
                    time_elapsed=curr_time_elapsed,
                    losses_dict=sum_losses,
                    iteration=iteration
                )
                logs.append(log)

            if iteration % 5000 == 0:
                save_model(
                    transformer=transformer_network,
                    optimizer=optimizer,
                    folder_path=model_folder_path,
                    iteration=iteration
                )

    # save last model
    save_model(
        transformer=transformer_network,
        optimizer=optimizer,
        folder_path=model_folder_path,
        iteration=iteration
    )

    # save dict to csv
    if not logs:
        raise ValueError("No logs were saved")

    columns_names = logs[0].keys()
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns_names)
        writer.writeheader()
        writer.writerows(logs)




if __name__ == "__main__":
    train(content_folder_path="images/content/train2017", style_img_name="starry_night.jpg")