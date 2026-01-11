import torch
import torch.nn as nn
from torchvision import models

class LossNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=weights)
        vgg16_features = vgg16.features
        #print(vgg16)
        self.slice1 = nn.Sequential(*vgg16_features[0:4])

        self.slice = nn.Sequential()
        for x in range(4):
            self.slice.add_module(str(x), vgg16_features[x])
        print(vgg16_features[0:4])    
        print("next")
        print(self.slice1)
        print("next")
        print(self.slice)