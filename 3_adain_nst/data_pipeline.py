# Helper functions for processing/loading data/images
import os
from typing import Tuple

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random


import numpy as np
import cv2 as cv

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 512
CROP_SIZE = 256
FLIP_PROB = 0.5

class TrainingDataset(Dataset):
    def __init__(self, content_dataset_path: str, style_dataset_path: str):
        self.content_dataset_path = Path(content_dataset_path)
        self.style_dataset_path = Path(style_dataset_path)

        self.content_imgs_paths = sorted(self.content_dataset_path.glob("*.jpg"))
        self.style_imgs_paths = sorted(self.style_dataset_path.glob("*.jpg"))
        print("COCO len: ", len(self.content_imgs_paths))
        print("WIKI len: ", len(self.style_imgs_paths))

        self.transform = TrainingDataset.get_transform_dataset()

    def __len__(self):
        return len(self.content_imgs_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        content_idx = idx
        style_idx = random.randint(0, len(self.style_imgs_paths) - 1)

        content_image = Image.open(self.content_imgs_paths[content_idx]).convert("RGB")
        style_image = Image.open(self.style_imgs_paths[style_idx]).convert("RGB")

        content_image = self.transform(content_image)
        style_image = self.transform(style_image)
        return content_image, style_image

    @staticmethod
    def get_transform_dataset():
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(p=FLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(VGG_MEAN, VGG_STD)])

# ##################################################################################################
# Invert
def get_img(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def get_img_from_frame(frame: np.ndarray) -> Image.Image:
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return img

def img_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    transform = get_transform()
    tr_img = transform(img)
    output = tr_img.unsqueeze(0).to(device) # (1, 3, H, W)
    return output

def get_transform() -> transforms.Compose:
    # transforms.Resize(IMG_SIZE),
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(VGG_MEAN, VGG_STD)])
    return transform

# Revert

def save_img(img: np.ndarray, output_path: str) -> None:
    ok = cv.imwrite(output_path, img)
    if not ok:
        raise RuntimeError(f"cv.imwrite failed for: {output_path}")
    print("Saved:", output_path)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    y = t.clamp(0.0, 1.0).cpu().numpy()
    y = (y * 255).astype("uint8")
    y = np.moveaxis(y, 0, 2) # CHW to HWC for openCV
    output_img = y[:, :, ::-1] # RGB to BGR  for openCV
    return output_img

def resize_img(img: Image.Image, target_size: int = 853) -> Image.Image:
    w, h = img.size

    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    return img.resize((new_w, new_h), Image.LANCZOS)