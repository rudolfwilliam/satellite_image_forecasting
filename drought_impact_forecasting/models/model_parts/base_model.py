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
        self.network = cfg["network"]
        print(self.network)
        print(type(self.network))

        conv_layer1 = self.__conv_layer_set(self.network[0][0], self.network[0][1])
        conv_layer2 = self.__conv_layer_set(self.network[1][0], self.network[1][1])

        fc1 = nn.Linear(self.network[2][0], self.network[2][1]) 
        fc2 = nn.Linear(self.network[3][0], self.network[3][1])
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

