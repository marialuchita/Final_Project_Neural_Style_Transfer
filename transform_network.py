# https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf#

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransormNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.create_downsampling()
        self.create_residual_blocks()
        self.create_upsampling()

        # output layer
        self.output_layer = ConvolutionalLayer(in_ch=32, out_ch=3, kernel_size=9, stride=1)

    def create_downsampling(self) -> None:
        self.conv1 = ConvolutionalLayer(in_ch=3, out_ch=32, kernel_size=9, stride=1)
        self.in_down1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvolutionalLayer(in_ch=32, out_ch=64, kernel_size=3, stride=2)
        self.in_down2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvolutionalLayer(in_ch=64, out_ch=128, kernel_size=3, stride=2)        
        self.in_down3 = nn.InstanceNorm2d(128, affine=True)

    def create_residual_blocks(self) -> None:
        blocks = [ResidualBlock(128) for _ in range(5)]
        self.residual_blocks = nn.Sequential(*blocks)
    
    def create_upsampling(self):
        self.upsample1 = UpsampleLayer(in_ch=128, out_ch=64)
        self.in_up1 = nn.InstanceNorm2d(64, affine=True)
        self.upsample2 = UpsampleLayer(in_ch=64, out_ch=32)
        self.in_up2 = nn.InstanceNorm2d(32, affine=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out_t = F.relu(self.in_down1(self.conv1(t)))
        out_t = F.relu(self.in_down2(self.conv2(out_t)))
        out_t = F.relu(self.in_down3(self.conv3(out_t)))

        out_t = self.residual_blocks(out_t)

        out_t = F.relu(self.in_up1(self.upsample1(out_t)))
        out_t = F.relu(self.in_up2(self.upsample2(out_t)))

        out_t = self.output_layer(out_t)
        return out_t

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding_mode="reflect") -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv_layer = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            padding_mode=padding_mode
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.conv_layer(t)

class ResidualBlock(nn.Module):
    def __init__(self, ch: int=128, kernel_size: int=3, stride: int=1) -> None:
        super().__init__()
        self.conv1 = ConvolutionalLayer(in_ch=ch, out_ch=ch, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvolutionalLayer(in_ch=ch, out_ch=ch, kernel_size=kernel_size, stride=stride)
        self.inst_norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.inst_norm2 = nn.InstanceNorm2d(ch, affine=True)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        res_t = t
        out_t = F.relu(self.inst_norm1(self.conv1(t)))
        out_t = self.inst_norm2(self.conv2(out_t)) 
        return out_t + res_t

class UpsampleLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, upsample_factor: int=2) -> None:
        super().__init__()
        self.conv = ConvolutionalLayer(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=1)
        self.upsample_factor = upsample_factor

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out_t = F.interpolate(t, scale_factor=self.upsample_factor, mode="nearest") # increase image resolution by copying the pixels nearby.
        out_t = self.conv(out_t)
        return out_t


