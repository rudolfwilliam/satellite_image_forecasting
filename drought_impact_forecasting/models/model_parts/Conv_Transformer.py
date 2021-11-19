import numpy as np

import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.net = nn.Sequential(
            nn.Linear(self.configs.input_length, self.configs.input_length*4),
            nn.LeakyReLU(),
            nn.Linear(self.configs.input_length*4, self.configs.input_length),
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):

    def __init__(self, configs):
        super(ConvAttention, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel=self.num_hidden[-1], out_channel=3*self.num_hidden[-1], kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel=self.num_hidden[-1], out_channel=self.num_hidden[-1], kernel_size=5, padding=2)
        )

    def forward(self, x, enc_out= None, dec=False):
        b, c, h, w, l = x.shape
        qkv_setlist = []
        Vout_list = []
        for i in l:
            qkv_setlist.append(self.conv1(x[..., i]))
        qkv_set = torch.stack(qkv_setlist, dim=-1)
        if dec:
            Q, K, V = torch.split(qkv_set, self.num_hidden[-1], dim=1)
        else:
            Q, K, _ = torch.split(qkv_set, self.num_hidden[-1], dim=1)
            V = enc_out

        for i in l:
            Qi = rearrange([Q[..., i]] * l + K, 'b n h w l -> (b l) n h w')
            tmp = rearrange(self.conv2(Qi), '(b l) n h w -> b n h w l', l=l)
            tmp = F.softmax(tmp, dim=4)                                       #(b, n, h, w, l)
            tmp = np.multiply(tmp, torch.stack([V[i]] * l, dim=-1))
            Vout_list.append(torch.sum(tmp, dim=4))                           #(b, n, h, w)
        Vout = torch.stack(Vout_list, dim=-1 )
        return Vout                                                           #(b, n, h, w, l)


class PositionalEncoding(nn.Module):

    def __init__(self, configs):
        super(PositionalEncoding, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            return_list = [torch.ones((self.configs.batch_size,
                                       self.configs.img_width,
                                       self.configs.img_width)).to(self.configs.device)*(position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden[-1])) for hid_j in range(self.num_hidden[-1])]
            return torch.stack(return_list, dim=1)

        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(self.configs.input_length)]
        sinusoid_table[0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.stack(sinusoid_table, dim=-1)

    def forward(self, x):
        '''
        :param x: (b, channel, h, w, seqlen)
        :return:
        '''
        return x + self.pos_table.clone().detach()


class Encoder(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.layers = nn.ModuleList([])
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        for _ in range(self.configs["depth"]):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 ConvAttention(self.configs))),
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 FeedForward(self.configs)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class Decoder(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        for _ in range(self.configs["depth"]):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 ConvAttention(self.configs))),
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 FeedForward(self.configs)))
            ]))

    def forward(self, x, enc_out, mask=None):
        for attn, ff in (self.layers):
            x = attn(x, enc_out=enc_out, dec=True)
            x = ff(x)
        return x


class Feature_Generator(nn.Module):
    def __init__(self, configs):
        super(Feature_Generator, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_hidden[0],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=self.num_hidden[0],
                               out_channels=self.num_hidden[1],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=self.num_hidden[1],
                               out_channels=self.num_hidden[2],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=self.num_hidden[2],
                               out_channels=self.num_hidden[3],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.bn1 = nn.BatchNorm2d(self.num_hidden[0])
        self.bn2 = nn.BatchNorm2d(self.num_hidden[1])
        self.bn3 = nn.BatchNorm2d(self.num_hidden[2])
        self.bn4 = nn.BatchNorm2d(self.num_hidden[3])

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope=0.01, inplace=False)
        return out


class Conv_Transformer(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.pos_embedding = PositionalEncoding(self.configs)
        self.Encoder = Encoder(self.dim, self.configs)
        self.Decoder = Decoder(self.dim, self.configs)
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], 1, kernel_size=1)
        )

    def forward(self, frames, num_pred, mask = None):
        b, n, h, w, l = frames.shape
        feature_map = self.feature_embedding(img=frames, configs=self.configs)
        enc_in = self.pos_embedding(feature_map)
        enc_out = self.Encoder(enc_in)
        # queries correspond to num_pred * (last embedding)
        dec_out = self.Decoder(enc_in[..., -1].repeat(num_pred), enc_out)
        out_list = []
        for i in l:
            out_list.append(self.back_to_pixel(dec_out[..., i]))
        x = torch.stack(out_list, dim=-1)

        return x

    def feature_embedding(self, img, configs):
        generator = Feature_Generator(configs).to(configs.device)
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[:, :, :, :, i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img
