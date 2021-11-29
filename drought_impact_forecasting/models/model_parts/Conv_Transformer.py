import numpy as np

import torch
from einops import rearrange
from .shared import Conv_Block
import torch.nn as nn
import torch.nn.functional as F



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
        return self.fn(torch.stack([self.norm(x[..., i]) for i in range(x.size()[-1])], dim=-1), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, configs, num_hidden):
        super().__init__()
        self.configs = configs
        self.num_hidden = num_hidden
        self.conv = Conv_Block(self.num_hidden, self.num_hidden, kernel_size=self.configs["kernel_size"],
                               dilation_rate=self.configs["dilation_rate"], num_conv_layers=self.configs["num_conv_layers"])

    def forward(self, x):
        return torch.stack([self.conv(x[..., i]) for i in range(x.size()[-1])], dim=-1)


class ConvAttention(nn.Module):
    def __init__(self, configs, num_hidden, enc):
        super(ConvAttention, self).__init__()
        self.configs = configs
        self.enc = enc
        self.num_hidden = num_hidden
        # important note: shared convolution is intentional here
        if self.enc:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=3 * self.num_hidden, kernel_size=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=2 * self.num_hidden, kernel_size=1)
            )
        #TODO: (optional) make this have multiple layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden*2, out_channels=1, kernel_size=5, padding=2)
        )

    def forward(self, x, enc_out=None):
        b, c, h, w, t = x.shape
        qkv_setlist = []
        if not self.enc:
            s = enc_out.size()[-1]
            for i in range(s):
                qkv_setlist.append(self.conv1(enc_out[..., i]))
        else:
            s = t
            for i in range(t):
                qkv_setlist.append(self.conv1(x[..., i]))

        qkv_set = torch.stack(qkv_setlist, dim=-1)
        if not self.enc:
            K, V = torch.split(qkv_set, self.num_hidden, dim=1)
            Q = x
        else:
            Q, K, V = torch.split(qkv_set, self.num_hidden, dim=1)

        K_rep = torch.stack([K] * t, dim=-1)
        V_rep = torch.stack([V] * t, dim=-1)
        Q_rep = torch.stack([Q] * s, dim=-1)
        K_flip = rearrange(K_rep, 'b c h w s t -> b c h w t s')
        Q_K = rearrange(torch.concat((Q_rep, K_flip), dim=1), 'b c h w t s -> (b t s) c h w') # no convolution across time dim!
        extr_feat = rearrange(torch.squeeze(self.conv2(Q_K)), '(b t s) h w -> b h w s t', b=b, t=t)
        attn_mask = F.softmax(extr_feat, dim=-2)
        V_pre = torch.stack([torch.mul(attn_mask, V_rep[:, c, ...]) for c in range(V_rep.size()[1])], dim=1)
        V_out = torch.sum(V_pre, dim=-2)

        return V_out


class PositionalEncoding(nn.Module):
    def __init__(self, configs):
        super(PositionalEncoding, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs["num_hidden"]

    def _get_sinusoid_encoding_table(self, t):
        ''' Sinusoid position encoding table '''
        # no differentiation should happen with the params in here!
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return_list = [torch.ones((self.configs["batch_size"],
                                       self.configs["img_width"],
                                       self.configs["img_width"])).to(self.configs["device"])*(position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden[-1])) for hid_j in range(self.num_hidden[-1])]
            return torch.stack(return_list, dim=1)

        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(t)]
        sinusoid_table = torch.stack(sinusoid_table, dim=0)
        sinusoid_table[:, :, 0::2] = np.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = np.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        return torch.moveaxis(sinusoid_table, 0, -1)

    def forward(self, x, t):
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(t))
        return x + self.pos_table.clone().detach()


class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.layers = nn.ModuleList([])
        self.num_hidden = self.configs["num_hidden"]
        for _ in range(self.configs["depth"]):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                 ConvAttention(self.configs, self.num_hidden[-1], enc=True))),
                Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                 FeedForward(self.configs, self.num_hidden[-1])))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.configs = configs
        self.num_hidden = self.configs["num_hidden"]
        self.num_non_pred_feat = self.configs["non_pred_channels"]
        for _ in range(self.configs["depth"]):
            if self.configs["query_self_attention"]:
                self.layers.append(nn.ModuleList([Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                     ConvAttention(self.configs, self.num_hidden[-1], enc=True)))]))
            self.layers.append(nn.ModuleList([
                # convolutional attention
                Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                 ConvAttention(self.configs, self.num_hidden[-1], enc=False))),
                # feed forward
                Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                 FeedForward(self.configs, self.num_hidden[-1])))
            ]))

    def forward(self, queries, enc_out):
        if self.configs["query_self_attention"]:
            for query_attn, attn, ff in self.layers:
                queries = query_attn(queries)
                x = attn(queries, enc_out=enc_out)
                x = ff(x)
        else:
            for attn, ff in self.layers:
                x = attn(queries, enc_out=enc_out)
                x = ff(x)

        return x


class Feature_Generator(nn.Module):
    def __init__(self, configs):
        super(Feature_Generator, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs["num_hidden"]
        self.conv1 = nn.Conv2d(in_channels=11,
                               out_channels=self.num_hidden[0],
                               kernel_size=self.configs["kernel_size"],
                               stride=1,
                               padding="same")
        self.conv2 = nn.Conv2d(in_channels=self.num_hidden[0],
                               out_channels=self.num_hidden[1],
                               kernel_size=self.configs["kernel_size"],
                               stride=1,
                               padding="same")
        self.conv3 = nn.Conv2d(in_channels=self.num_hidden[1],
                               out_channels=self.num_hidden[2],
                               kernel_size=self.configs["kernel_size"],
                               stride=1,
                               padding="same")
        self.conv4 = nn.Conv2d(in_channels=self.num_hidden[2],
                               out_channels=self.num_hidden[3],
                               kernel_size=self.configs["kernel_size"],
                               stride=1,
                               padding="same")
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
        self.num_hidden = configs["num_hidden"]
        self.pos_embedding = PositionalEncoding(self.configs)
        self.Encoder = Encoder(self.configs)
        self.Decoder = Decoder(self.configs)
        # make the dimensions of the non_pred features fit
        self.dim_fit = Conv_Block(self.configs["non_pred_channels"], self.num_hidden[-1], num_conv_layers=1, kernel_size=configs["kernel_size"])
        # last predictions needs a dummy input
        self.blank = torch.stack([torch.zeros(size=(configs["batch_size"], configs["non_pred_channels"],
                                                    self.configs["img_width"], self.configs["img_width"]))], dim=-1)
        #TODO: replace this by SFFN
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], 4, kernel_size=1)
        )

    def forward(self, frames, prediction_count, non_pred_feat = None):
        _, _, _, _, t = frames.size()
        feature_map = self.feature_embedding(img=frames, configs=self.configs)
        enc_in = self.pos_embedding(feature_map, t)
        enc_out = self.Encoder(enc_in)
        # queries correspond to non_pred_feat
        if non_pred_feat is not None:
            non_pred_feat = torch.concat((self.blank, non_pred_feat), dim=-1)
        else:
            non_pred_feat = self.blank

        dec_V = torch.stack([self.dim_fit(non_pred_feat[..., i]) for i in range(non_pred_feat.size()[-1])], dim=-1)
        dec_out = self.Decoder(dec_V, enc_out)
        out_list = []
        for i in range(prediction_count):
            out_list.append(self.back_to_pixel(dec_out[..., i]))
        x = torch.stack(out_list, dim=-1)

        return x

    def feature_embedding(self, img, configs):
        generator = Feature_Generator(configs).to(configs["device"])
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[..., i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img
