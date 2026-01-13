import torch
import torch.nn as nn
from torchvision import models

class LossNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=weights).eval()
        vgg16_features = vgg16.features

        features1 = vgg16_features[0:4]
        self.group1 = nn.Sequential(*features1)
        features2 = vgg16_features[4:9]
        self.group2 = nn.Sequential(*features2)
        features3 = vgg16_features[9:16]
        self.group3 = nn.Sequential(*features3)
        features4 = vgg16_features[16:23]
        self.group4 = nn.Sequential(*features4)

        # freeze the parameters
        parameters = self.parameters()
        for parameter in parameters:
            parameter.requires_grad_(False)


    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        :param t: Tensor of shape (N, 3, H, W)
        :return: VGG feature maps dictionary
        """
        output = self.group1(t)
        relu1_2 = output

        output = self.group2(output)
        relu2_2 = output

        output = self.group3(output)
        relu3_3 = output

        output = self.group4(output)
        relu4_3 = output

        return {
            "relu1_2": relu1_2,
            "relu2_2": relu2_2,
            "relu3_3": relu3_3,
            "relu4_3": relu4_3
        }


