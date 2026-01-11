from loss_network import LossNetwork
from transform_network import TransormNetwork

import torch
from torch.optim import Adam
import numpy as np

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
