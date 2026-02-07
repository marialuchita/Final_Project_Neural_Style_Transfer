import os
import torch
import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import transforms
from transform_network import TransformNetwork
from datetime import datetime

CONTENT_IMAGE = "images/content/puppy.jpg"
MODEL_PATH    = "models/20260204_125032_sunset/model_1_29570.pth"   # <-- update to your new saved name
OUTPUT_IMAGE  = "output_images"
USE_CUDA      = True
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

@torch.no_grad()
def main():
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TransformNetwork().to(device).eval()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["transformer_state_dict"], strict=True)

    img = Image.open(CONTENT_IMAGE).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize(VGG_MEAN, VGG_STD)])
    x = transform(img).unsqueeze(0).to(device)  # (1,3,H,W) in [0,1]

    y = model(x)[0]  # output in normalized ImageNet space

    # de-normalize to [0, 1] RGB
    mean = torch.tensor(VGG_MEAN, device=y.device).view(3, 1, 1)
    std = torch.tensor(VGG_STD, device=y.device).view(3, 1, 1)
    y = (y * std + mean).clamp(0.0, 1.0)
    y = y.cpu().numpy()
    y = (y * 255).astype(np.uint8)
    y = np.moveaxis(y, 0, 2)  # CHW -> HWC (RGB)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{OUTPUT_IMAGE}/{timestamp}.jpg"

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    cv.imwrite(out_dir, y[:, :, ::-1])  # RGB -> BGR
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
