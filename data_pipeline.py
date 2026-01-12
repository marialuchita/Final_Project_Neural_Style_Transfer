from PIL import Image
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

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
    batch_size = 4
    workers = 2
    dataset = TrainDataset(folder_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, pin_memory=True)
    return loader

def normalize_for_vgg(t: torch.Tensor) -> torch.Tensor:
    t_mean = t.new_tensor(VGG_MEAN).view(1, 3, 1, 1) # N - batch size, C - number of channels (RGB), H - height, W - width
    t_std = t.new_tensor(VGG_STD).view(1, 3, 1, 1)
    out_t = (t - t_mean) / t_std
    return out_t


