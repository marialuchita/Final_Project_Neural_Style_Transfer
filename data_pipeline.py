from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        self.images_paths = [
            path for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ] #https://docs.python.org/3/library/pathlib.html

        #print(self.images_paths)
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
            transforms.ToTensor()])
        return t

def normalize_for_vgg(t: torch.Tensor) -> torch.Tensor:
    # t should have the shape (N, 3, H, W)
    t_mean = t.new_tensor(VGG_MEAN).view(1, 3, 1, 1) # N - batch size, C - number of channels (RGB), H - height, W - width
    t_std = t.new_tensor(VGG_STD).view(1, 3, 1, 1)
    out_t = (t - t_mean) / t_std
    return out_t

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

