import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from torchvision import models, transforms
from torch.utils.data import DataLoader

from style_transfer_network import StyleTransferNet, VGGEncoder, VGGDecoder
from data_pipeline import TrainingDataset

LEARNING_RATE = 1e-4
EPOCHS = 2

BATCH_SIZE = 4
WORKERS = 6

COCO_PATH = ""
WIKI_PATH = ""

def compute_content_loss(content_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(content_features, target_features)

def compute_style_loss(style_features:  torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    style_mean, style_std = StyleTransferNet.calc_statistics(style_features)
    target_mean, target_std = StyleTransferNet.calc_statistics(target_features)
    return F.mse_loss(style_mean, target_mean) + F.mse_loss(style_std, target_std)
    # style_loss = torch.tensor(0.0, device=generated_img_features[0].device)
    # features_zip = zip(generated_img_features, style_img_features)
    # for gen_img_f, st_img_f in features_zip:
    #     gen_img_mean, gen_img_std = AdaINLayer.calc_statistics(gen_img_f)
    #     st_img_mean, st_img_std = AdaINLayer.calc_statistics(st_img_f)
    #     style_loss += (F.mse_loss(gen_img_mean, st_img_mean) +
    #                   F.mse_loss(gen_img_std, st_img_std))
    # return style_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    network = StyleTransferNet().to(device)

    # Freeze the encoder
    network.encoder.eval()
    for parameter in network.encoder.parameters():
        parameter.requires_grad_(False)
    # Make the decoder trainable
    network.decoder.train()
    for parameter in network.decoder.parameters():
        parameter.requires_grad_(True)

    optimizer = Adam(network.decoder.parameters(), lr=LEARNING_RATE)

    dataset = TrainingDataset(COCO_PATH, WIKI_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    for epoch in range(EPOCHS):




if __name__ == "__main__":



