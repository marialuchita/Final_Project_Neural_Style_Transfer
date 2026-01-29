import argparse
import os
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from data_pipeline import process_image, denormalize, save_as_image
from style_transfer_network import StyleTransferNet

ALPHA = 1.0

@torch.no_grad()
def stylise(content_img_path: str, style_img_path: str, model_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = process_image(content_img_path, device)
    style_img = process_image(style_img_path, device)

    model = torch.load(model_path, map_location=device)
    network = StyleTransferNet().to(device).eval()
    network.decoder.load_state_dict(model["state_dict"], strict=True)

    # forward pass
    output_network = network(content_img=content_img, style_img=style_img, alpha=ALPHA)
    denormalized_output = denormalize(output_network)
    save_as_image(denormalized_output, output_path)

def main():
    content_img_path = "images/content/puppy.jpg"
    style_img_path = "images/wikiart/albrecht-durer_crucifixion-1498.jpg"
    model_path = "models/model_2_29570.pth"
    output_path = "outputs/stylized.png"
    stylise(
        content_img_path=content_img_path,
        style_img_path=style_img_path,
        model_path=model_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()