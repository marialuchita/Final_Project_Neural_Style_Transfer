import torch
import torch.nn as nn
from torchvision import models

from typing import Union, List

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT
        vgg19 = models.vgg19(weights=weights).features

        # according to the paper: https://arxiv.org/pdf/1703.06868
        # till layer relu4_1
        self.group1 = nn.Sequential(*vgg19[0:2]) # up to and including relu1_1
        self.group2 = nn.Sequential(*vgg19[2:7]) # up to and including relu2_1
        self.group3 = nn.Sequential(*vgg19[7:12]) # up to and including relu3_1
        self.group4 = nn.Sequential(*vgg19[12:21]) # up to and including relu4_1

        # freeze the parameters
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, return_all: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        :param t: Tensor of shape (N, 3, H, W)
        :return: Union[torch.Tensor, List[torch.Tensor]]
        """
        relu1_1_features = self.group1(x)
        relu2_1_features = self.group2(relu1_1_features)
        relu3_1_features = self.group3(relu2_1_features)
        relu4_1_features = self.group4(relu3_1_features)

        if return_all:
            return [relu1_1_features, relu2_1_features, relu3_1_features, relu4_1_features]
        return relu4_1_features



class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

