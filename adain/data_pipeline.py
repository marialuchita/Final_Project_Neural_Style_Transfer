from typing import Tuple

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
from typing import Tuple

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

        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(p=FLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(VGG_MEAN, VGG_STD)])

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
