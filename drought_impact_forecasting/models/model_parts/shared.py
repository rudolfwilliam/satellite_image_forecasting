import torch
from torch import nn
import torchvision

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=3, dilation_rate=2):
        super(Conv_Block, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.input_dim = in_channels
        self.output_dim = out_channels

        ops = []
        for i in range(self.num_conv_layers):
            ops.append(nn.Conv2d(in_channels, in_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=False, padding='same'))
            ops.append(nn.BatchNorm2d(in_channels))
            ops.append(nn.ReLU())
        ops.append(nn.Conv2d(in_channels, out_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same'))
        self.seq = nn.Sequential(*ops)

    def forward(self, input_tensor): # irrelevant?
        out = self.seq(input_tensor)
        return out

class Contractor(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.blocks = []
        self.pools = []
        for i in range(len(channels)-1):
            self.blocks.append(Conv_Block(channels[i], channels[i+1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
            self.pools.append(nn.MaxPool2d(2))
        #self.seq = nn.Sequential(*ops)

    def forward(self, x):
        outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outputs.append(x)
            x = self.pools[i](x)
        return outputs

class Expandor(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        ops = []
        for i in range(len(channels)-1):
            self.up_convs.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2))
            self.blocks.append(Conv_Block(channels[i+1], channels[i+1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
            #ops.append(Conv_Block(channels[i], channels[i+1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
            #ops.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2))
        #self.seq = nn.Sequential(*ops)

    def forward(self, x, features):
        for i in range(len(self.up_convs)-1):
            # Check that H,W are at end
            cur_features = features[-(i+1)] # Possibly move this cropping inside constructor
            cur_features = torchvision.transforms.CenterCrop(features.shape[-2:])(features[-(i+1)])
            x = self.up_convs(x)
            x = torch.cat([x, cur_features], dim=1) # check dim
            x = self.dec_blocks[i](x)
        return x

class U_Net(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate=1):
        super(U_Net, self).__init__()
        self.num_conv_layers = len(channels)
        self.input_dim = channels[0]
        self.output_dim = channels[-1]
        
        steps = round(len(channels)/2)
        net = nn.ModuleList()
        net.append(Contractor(channels[:steps], kernel_size=kernel_size, dilation_rate=dilation_rate))
        net.append(Conv_Block(channels[steps-1], channels[steps], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
        net.append(Expandor(channels[steps:], kernel_size=kernel_size, dilation_rate=dilation_rate))
        #ops.append(nn.Conv2d(channels[-2], channels[-1], dilation=dilation_rate, num_conv_layers=2, kernel_size=kernel_size,
        #                             bias=True, padding='same'))
        
    def forward():
        pass