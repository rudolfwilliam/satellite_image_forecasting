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
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

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

