from PIL import Image
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainDataset(Dataset):
    """
    Dataset class for training data

    """
    def __init__(self, folder_path: str):
        self.dataset_dir = Path(folder_path)
        self.images_paths = [
            path for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ] #https://docs.python.org/3/library/pathlib.html

        self.transform = self.get_transform()

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path =  self.images_paths[index]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor

    def get_transform(self) -> transforms.Compose:
        img_size = 256
        flip_prob = 0.5
        t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ToTensor()])
        return t

def data_loader(folder_path: str):
    dataset = TrainDataset(folder_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    return loader