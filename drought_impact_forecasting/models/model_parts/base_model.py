import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import math
import matplotlib.pyplot as plt


# The encoder will use LeakyReLU!
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        pass

    def forward(self):
        pass

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        pass

    def forward(self):
        pass

# The 2 Discriminators share the same architecture, but not the weights
# 3D convolutional neural net
class Discriminator_VAE(torch.nn.Module):
    def __init__(self, cfg):
        pass

    def forward(self):
        pass

class Discriminator_GAN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.classes = cfg["model"]["classes"]
        # TODO: Parametrize everything using config/base_model.json. No numbers must be here!
        conv_layer1 = self.__conv_layer_set(3, 32)
        conv_layer2 = self.__conv_layer_set(32, 64)

        fc1 = nn.Linear(64*28*28*28, 2) 
        fc2 = nn.Linear(1404928, self.classes)
        relu = nn.LeakyReLU()
        
        self.layers = [conv_layer1, conv_layer2, fc1, fc2, relu]
        self.net = torch.nn.Sequential(*self.layers)

    def __conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        return F.softmax(self.net(x))

