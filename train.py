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
LEARNING_RATE = 0.001
EPOCHS = 2

CONTENT_LAYER = "relu2_2"
STYLE_LAYERS = "relu2_2"
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 4e5 # Johnson uses 1e5 to 4e5
TV_WEIGHT = 0 # or 1e-6 to 1e-4



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

def train(content_folder_path: str, style_img_path: str):
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


    for epoch in range(EPOCHS):
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
            content_loss = F.mse_loss(content_features[CONTENT_LAYER], style_features[CONTENT_LAYER])

            # -------------------------------------------------------------------------
            # Style loss >>>
            # -------------------------------------------------------------------------
            # MSE of the gram matrices of the feature maps of the style image
            # and the feature maps of the generated/stylized image
            style_loss = 0
            for layer, gram_m in style_gram_matrices.items():
                current_mse = F.mse_loss(gram_m, target_style_gram_matrices[layer])
                style_loss += current_mse
            style_loss /= len(style_gram_matrices)

            # -------------------------------------------------------------------------
            # Total variation loss
            # -------------------------------------------------------------------------
            tv_loss = compute_tv_loss(stylized_batch)

            # -------------------------------------------------------------------------
            # Total loss
            # -------------------------------------------------------------------------
            total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss

            optimizer.zero_grad() # clear the old gradients
            total_loss.backward() # compute new gradients
            optimizer.step() # update new weights

            if batch_index % 100 == 0:
                print(f"Epoch: {epoch + 1} | ",
                      f"Content loss: {content_loss.item():.4f} | ",
                      f"Style loss: {style_loss:.4f} | ",
                      f"TV loss: {tv_loss.item():.4f} | ",
                      f"Total loss: {total_loss.item():.4f}")




if __name__ == "__main__":
    train(content_folder_path="images/content/train2017", style_img_path="images/style/starry_night.jpg")