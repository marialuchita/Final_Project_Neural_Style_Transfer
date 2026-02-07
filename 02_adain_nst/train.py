import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from torchvision import models, transforms
from torch.utils.data import DataLoader

from style_transfer_network import *
from data_pipeline import TrainingDataset

import os
import csv
import time
from datetime import datetime

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

LEARNING_RATE = 1e-4
EPOCHS = 4

BATCH_SIZE = 4
WORKERS = 6

COCO_PATH = "../images/coco/train2017"
WIKI_PATH = "images/wikiart"

MODELS_PATH = "models"
LOGS_PATH = "logs"
ALPHA = 1.0

WEIGHTS = {
    "content": 1.0,
    "style": 3.0,
}

def compute_content_loss(content_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(content_features, target_features)

def compute_style_loss(style_features:  torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    style_mean, style_std = calc_statistics(style_features)
    target_mean, target_std = calc_statistics(target_features)
    return F.mse_loss(style_mean, target_mean) + F.mse_loss(style_std, target_std)

def save_model(module: nn.Module, folder_path: str, epoch: int, batch: int) -> None:

    out_model_path = os.path.join(folder_path, f"model_{epoch}_{batch}.pth")
    torch.save(
        {"state_dict": module.state_dict()},
        out_model_path
    )

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_logs = os.path.join(LOGS_PATH, f"{timestamp}_logs.csv")
    csv_weights = os.path.join(LOGS_PATH, f"{timestamp}_weights.csv")

    # StyleTransferNet
    network = StyleTransferNet().to(device)

    # Freeze the encoder
    network.encoder.eval()
    for parameter in network.encoder.parameters():
        parameter.requires_grad_(False)

    # Make the decoder trainable
    network.decoder.train()
    for parameter in network.decoder.parameters():
        parameter.requires_grad_(True)

    # optimizer
    optimizer = Adam(network.decoder.parameters(), lr=LEARNING_RATE)

    # TrainingDataset and DataLoader

    dataset = TrainingDataset(COCO_PATH, WIKI_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    logs = []
    iteration = 0
    start_time = time.time()
    curr_time_elapsed = 0

    tr = transforms.Normalize(VGG_MEAN, VGG_STD)

    for epoch in range(EPOCHS):
        network.encoder.eval()
        network.decoder.train()
        total_loss = 0
        for batch_idx, (content_image, style_image) in enumerate(loader):
            content_image = content_image.to(device)
            style_image = style_image.to(device)

            # build Adain target (without creating encoder graph)

            with torch.no_grad():
                cont_feat = network.encoder(content_image)
                style_feat = network.encoder(style_image)
                target_feat = adain(cont_feat, style_feat)
                target_feat = ALPHA * target_feat + (1 - ALPHA) * cont_feat

            # trainable decoder - decode to img
            decoder_output = network.decoder(target_feat)
            decoder_output_n = tr(decoder_output)

            # features of decoder output and style img
            decoder_output_cont_f = network.encoder(decoder_output_n)
            decoder_output_style_f = network.encoder(decoder_output_n, return_all=True)

            with torch.no_grad():
                style_features = network.encoder(style_image, return_all=True)

            # COMPUTE LOSSES:
            # content loss
            content_loss = compute_content_loss(decoder_output_cont_f, target_feat)

            # style loss
            style_loss = 0
            style_zip = zip(decoder_output_style_f, style_features)
            for sf, tf in style_zip:
                style_loss += compute_style_loss(sf, tf)

            # current total loss
            current_loss = WEIGHTS["content"] * content_loss + WEIGHTS["style"] * style_loss

            # Backpropagate:
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            iteration += 1
            curr_time_elapsed = (time.time() - start_time) / 60 # in minutes

            if batch_idx % 100 == 0 or (epoch == EPOCHS - 1 and batch_idx == len(loader) - 1 ) :
                log = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "content_loss": content_loss.item(),
                    "style_loss": style_loss.item(),
                    "total_loss": current_loss.item(),
                    "time_minutes": curr_time_elapsed,
                }
                print(log)
                logs.append(log)

            if batch_idx % 3000 == 0 or (epoch == EPOCHS - 1 and batch_idx == len(loader) - 1 ):
                save_model(
                    module=network.decoder,
                    folder_path=MODELS_PATH,
                    epoch=epoch + 1,
                    batch=batch_idx,
                )

    # save dict to csv
    if not logs:
        raise ValueError("No logs were saved")

    columns_names = logs[0].keys()
    with open(csv_logs, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns_names)
        writer.writeheader()
        writer.writerows(logs)



    with open(csv_weights, mode='w', newline='') as file2:
        writer = csv.DictWriter(file2, fieldnames=WEIGHTS.keys())
        writer.writeheader()
        writer.writerow(WEIGHTS)

if __name__ == "__main__":
    train()

