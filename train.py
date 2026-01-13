from loss_network import *
from transform_network import TransormNetwork


import torch

import numpy as np
from data_pipeline import *
from torch.utils.data import DataLoader
from torch.optim import Adam

BATCH_SIZE = 4
WORKERS = 2
LEARNING_RATE = 0.001

def process_style_img(style_img_path: str, device: torch.device) -> torch.Tensor:
    style_img = Image.open(style_img_path).convert("RGB")
    transform = transforms.ToTensor() # transforms to 0 - 1
    image_tensor = transform(style_img).unsqueeze(0).to(device) # 1, C, H, W
    image_tensor = image_tensor.repeat(BATCH_SIZE, 1, 1, 1)  # B, C, H, W
    return image_tensor

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

    transformer_network = TransormNetwork().train().to(device)
    vgg_network = LossNetwork().to(device)
    vgg_network.eval()
    vgg_network.requires_grad_(False)
    optimizer = torch.optim.Adam(transformer_network.parameters(), lr=LEARNING_RATE)
    style_img_t = process_style_img(style_img_path, device)

    with torch.no_grad():
        style_img_features = vgg_network(normalize_for_vgg(style_img_t))
        target_style_gram_matrices = [gram_matrix(f) for f in style_img_features.values()]


    for i, batch in enumerate(loader):
        print(f"batch {i}: type={type(batch)}, shape={batch.shape}")
        g = gram_matrix(batch)
        print(f"gram {i}: type={type(g)}, shape={g.shape}")
        #print(batch.shape)
        #if i == 0:
        #    print(batch)
        #print(len(batch))
        #print(type(batch))
        #print([type(b) for b in batch])
        #x, y = batch
        #x = x.to("cuda", non_blocking=True) if torch.cuda.is_available() else x
        #y = y.to("cuda", non_blocking=True) if torch.cuda.is_available() else y
        print("")

if __name__ == "__main__":
    train(content_folder_path="images/content", style_img_path="images/style/starry_night.jpg")