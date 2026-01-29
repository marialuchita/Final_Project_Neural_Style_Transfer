import os.path
from typing import Tuple

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
from typing import Tuple

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

def process_image(path: str, device: torch.device) -> torch.Tensor:
    transform = get_transform()
    img = Image.open(path).convert("RGB")
    tr_img = transform(img)
    output = tr_img.unsqueeze(0).to(device) # (1, 3, H, W)
    return output

def process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Converts a single video frame into a tensor for the trained model.
    :param frame:
    :param device:
    :return:
    """
    transform = get_transform()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    tr_img = transform(img)
    output = tr_img.unsqueeze(0).to(device) # (1, 3, H, W)
    return output

def get_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(VGG_MEAN, VGG_STD)])
    return transform

def denormalize(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(0)
    std = torch.tensor(VGG_STD, device=img.device).view(1, 3, 1, 1)
    mean = torch.tensor(VGG_MEAN, device=img.device).view(1, 3, 1, 1)
    output = (img * std + mean).clamp(0, 1)
    return output

def save_as_image(img: torch.Tensor, output_path: str) -> None:
    if img.dim() == 4:
        img = img[0]
    y = img.clamp(0.0, 1.0).cpu().numpy()
    y = (y * 255).astype("uint8")
    y = np.moveaxis(y, 0, 2) # CHW to HWC
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, y[:, :, ::-1]) # RGB to BGR



