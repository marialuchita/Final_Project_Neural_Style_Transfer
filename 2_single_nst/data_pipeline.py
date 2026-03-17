# Helper functions for processing/loading data/images
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2 as cv

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 256
FLIP_PROB = 0.5

class TrainDataset(Dataset):
    """
    Dataset class for training data

    """
    def __init__(self, folder_path: str):
        self.dataset_dir = Path(folder_path)
        #print("dataset init ", self.dataset_dir)
        self.images_paths = sorted(self.dataset_dir.glob("*.jpg")) # use to read fast the paths. File format for coco dataset is .jpg
        #print("number of paths",len(self.images_paths))
        
        #print(self.images_paths[:10])
        self.transform = self.get_transform()

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path =  self.images_paths[index]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor

    @staticmethod
    def get_transform() -> transforms.Compose:
        t = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=FLIP_PROB),
            transforms.ToTensor(), 
            transforms.Normalize(VGG_MEAN, VGG_STD)])
        return t



def gram_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    :param t: Tensor of shape (N, C, H, W)
    :return: Tensor of shape (N, C, C)
    """
    n, c, h, w = t.shape
    features = t.view(n, c, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram = gram / (c * h * w)
    return gram

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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(VGG_MEAN, VGG_STD)])
    return transform

# Revert

def save_img(img: np.ndarray, output_path: str) -> None:
    ok = cv.imwrite(output_path, img)
    if not ok:
        raise RuntimeError(f"cv.imwrite failed for: {output_path}")
    print("Saved:", output_path)

def denormalize(t: torch.Tensor):
    mean = torch.tensor(VGG_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(VGG_STD, device=t.device).view(3, 1, 1)
    y = (t * std + mean).clamp(0.0, 1.0).cpu().numpy()
    y = (y * 255).astype("uint8")
    return y

def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    output = denormalize(t)
    output_img = np.moveaxis(output, 0, 2) # CHW to HWC for openCV
    output_img = output_img[:, :, ::-1] # RGB to BGR  for openCV
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