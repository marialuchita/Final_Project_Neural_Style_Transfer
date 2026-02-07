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

# def normalize_for_vgg(t: torch.Tensor) -> torch.Tensor:
#     # t should have the shape (N, 3, H, W)
#     t_mean = t.new_tensor(VGG_MEAN).view(1, 3, 1, 1) # N - batch size, C - number of channels (RGB), H - height, W - width
#     t_std = t.new_tensor(VGG_STD).view(1, 3, 1, 1)
#     out_t = (t - t_mean) / t_std
#     return out_t

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
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(VGG_MEAN, VGG_STD)])
    return transform

def save_as_image(y: torch.Tensor, output_path: str) -> None:
    mean = torch.tensor(VGG_MEAN, device=y.device).view(3, 1, 1)
    std = torch.tensor(VGG_STD, device=y.device).view(3, 1, 1)
    y = (y * std + mean).clamp(0.0, 1.0)
    y = y.cpu().numpy()
    y = (y * 255).astype(np.uint8)
    y = np.moveaxis(y, 0, 2)  # CHW -> HWC (RGB)
    cv.imwrite(output_path, y[:, :, ::-1])  # RGB -> BGR
    print("Saved:", output_path)



def tensor_to_frame(img: torch.Tensor) -> np.ndarray:
    if img.dim() == 4:
        img = img[0]
    mean = torch.tensor(VGG_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(VGG_STD, device=img.device).view(3, 1, 1)
    y = (img * std + mean).clamp(0.0, 1.0).cpu().numpy()
    y = (y * 255).astype("uint8")
    y = np.moveaxis(y, 0, 2) # CHW to HWC for openCV
    y = y[:, :, ::-1] # RGB to BGR  for openCV
    return y
