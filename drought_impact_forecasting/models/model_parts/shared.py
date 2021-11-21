from torch import nn

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  num_conv_layers=3, dilation_rate=2):
        super(Conv_Block, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.input_dim = in_channels
        self.output_dim = out_channels

        # define operations
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        # bias not needed due to batch norm
        self.in_mid_conv = nn.Conv2d(in_channels, in_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=False, padding='same')
        self.out_conv = nn.Conv2d(in_channels, out_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same')
    def forward(self, input_tensor):
        ops = []
        for i in range(self.num_conv_layers):
            ops.append(self.in_mid_conv)
            ops.append(self.norm)
            ops.append(self.relu)
        ops.append(self.out_conv)
        seq = nn.Sequential(*ops)
        out = seq(input_tensor)
        return out

