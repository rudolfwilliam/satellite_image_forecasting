import torch
import torchvision
import torch.nn.functional as F
from torch import nn

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=3, dilation_rate=2):
        super(Conv_Block, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.input_dim = in_channels
        self.output_dim = out_channels

        ops = []
        for i in range(self.num_conv_layers):
            ops.append(nn.Conv2d(in_channels, in_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=False, padding='same', padding_mode='reflect'))
            ops.append(nn.BatchNorm2d(in_channels))
            ops.append(nn.ReLU())
        ops.append(nn.Conv2d(in_channels, out_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same', padding_mode='reflect'))
        self.seq = nn.Sequential(*ops)

    def forward(self, input_tensor):
        out = self.seq(input_tensor)
        return out

class Contractor(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.pools = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.pools.append(nn.MaxPool2d(2))
            self.blocks.append(Conv_Block(channels[i], channels[i+1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for i in range(len(self.blocks)):
            x = self.pools[i](x)
            outputs.append(x)
            x = self.blocks[i](x)
        return outputs

class Expandor(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.up_convs.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2))
            self.blocks.append(Conv_Block(channels[i], channels[i+1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))

    def forward(self, x):
        cur = x[-1]
        for i in range(len(self.up_convs)):
            cur_features = x[-(i+2)] # Possibly move this cropping inside constructor
            cur_features = torchvision.transforms.CenterCrop([cur_features.shape[-2],cur_features.shape[-1]])(cur_features)
            cur = self.up_convs[i](cur)
            cur = torch.cat([cur, cur_features], dim=1) # check dim
            cur = self.blocks[i](cur)
        return cur

class U_Net(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate=1):
        super(U_Net, self).__init__()
        self.num_conv_layers = len(channels)
        self.input_dim = channels[0]
        self.output_dim = channels[-1]
        
        steps = round(len(channels)/2)
        self.net = nn.ModuleList()
        self.net.append(Conv_Block(channels[0], channels[1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
        self.net.append(Contractor(channels[1:steps+1], kernel_size=kernel_size, dilation_rate=dilation_rate))
        self.net.append(Expandor(channels[steps:-1], kernel_size=kernel_size, dilation_rate=dilation_rate))
        self.net.append(Conv_Block(channels[-2], channels[-1], kernel_size, num_conv_layers=2, dilation_rate=dilation_rate))
        
    def forward(self, x):
        x = self.net[0](x)
        c = self.net[1](x)
        e = self.net[2](c)
        o = self.net[3](e)
        o = F.interpolate(o, x.shape[-1]) # get back the right dims we started with
        return o