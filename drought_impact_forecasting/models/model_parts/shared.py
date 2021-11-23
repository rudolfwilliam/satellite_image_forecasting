from torch import nn

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  num_conv_layers=3, dilation_rate=2):
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

    def forward(self, input_tensor):
        out = self.seq(input_tensor)
        return out

