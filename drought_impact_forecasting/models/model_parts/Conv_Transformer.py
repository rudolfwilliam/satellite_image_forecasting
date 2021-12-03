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
    def __init__(self, configs, num_hidden, enc, mask=False):
        super(ConvAttention, self).__init__()
        self.configs = configs
        self.enc = enc
        self.mask = mask
        self.num_hidden = num_hidden
        # important note: shared convolution is intentional here
        if self.enc:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=3 * self.num_hidden, kernel_size=1, padding="same")
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=2 * self.num_hidden, kernel_size=1, padding="same")
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden*2, out_channels=1, kernel_size=self.configs["kernel_size"], padding="same")
        )

    def forward(self, x, enc_out=None):
        b, c, h, w, s = x.shape
        if self.enc:
            qkv_setlist = []
            t = s
            for i in range(t):
                qkv_setlist.append(self.conv1(x[..., i]))
            qkv_set = torch.stack(qkv_setlist, dim=-1)
            Q, K, V = torch.split(qkv_set, self.num_hidden, dim=1)
        else:
            # x correspond to the query
            kv_setlist = []
            t = enc_out.size()[-1]
            for i in range(t):
                kv_setlist.append(self.conv1(enc_out[..., i]))
            kv_set = torch.stack(kv_setlist, dim=-1)
            K, V = torch.split(kv_set, self.num_hidden, dim=1)
            Q = x

        K_rep = torch.stack([K] * s, dim=-1)
        V_rep = torch.stack([V] * s, dim=-1)
        Q_rep = torch.stack([Q] * t, dim=-1)
        K_flip = rearrange(K_rep, 'b c h w t s -> b c h w s t')
        Q_K = torch.concat((Q_rep, K_flip), dim=1) 
        if self.mask:
            # only feed in 'previous' keys & values for computing softmax
            V_out = []
            for i in range(t):
                Q_K_temp = Q_K[..., :i+1, i]
                Q_K_temp = rearrange(Q_K_temp, 'b c h w t-> (b t) c h w') # no convolution across time dim!
                extr_feat = rearrange(torch.squeeze(self.conv2(Q_K_temp)), '(b t) h w -> b h w t', b=b, t=i+1)
                attn_mask = F.softmax(extr_feat, dim=-1)
                V_pre = torch.stack([torch.mul(attn_mask, V_rep[:, c, :, :, i, :i+1]) for c in range(V_rep.size()[1])], dim=1)
                V_out.append(torch.sum(V_pre, dim=-1))
            V_out = torch.stack(V_out, dim=-1)
        else:
            Q_K = rearrange(Q_K, 'b c h w s t -> (b s t) c h w') # no convolution across time dim!
            extr_feat = rearrange(torch.squeeze(self.conv2(Q_K)), '(b s t) h w -> b h w t s', b=b, t=t)
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
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # even dim
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # odd dim

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
            self.layers.append(nn.ModuleList([
                # (masked) query self-attention
                Residual(PreNorm([self.num_hidden[-1], self.configs["img_width"], self.configs["img_width"]],
                                     ConvAttention(self.configs, self.num_hidden[-1], enc=True, mask=True))),
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
            for dec, attn, ff in self.layers:
                x = dec(queries)
                x = attn(queries, enc_out=enc_out)
                x = ff(x)

        return x

class Conv_Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_hidden = configs["num_hidden"]
        self.pos_embedding = PositionalEncoding(self.configs)
        self.Encoder = Encoder(self.configs)
        self.Decoder = Decoder(self.configs)
        self.input_feat_gen = Conv_Block(self.configs["in_channels"], self.num_hidden[-1], num_conv_layers=configs["num_layers_input_feat"], kernel_size=configs["kernel_size"])
        # make the dimensions of the non_pred features fit
        self.query_feat_gen = Conv_Block(self.configs["non_pred_channels"], self.num_hidden[-1], num_conv_layers=configs["num_layers_query_feat"], kernel_size=configs["kernel_size"])
        # last predictions needs a dummy input
        self.blank = torch.stack([torch.zeros(size=(configs["batch_size"], configs["non_pred_channels"],
                                                    self.configs["img_width"], self.configs["img_width"]))], dim=-1)
        #TODO: replace this by SFFN
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], 4, kernel_size=1)
        )

    def forward(self, frames, prediction_count, non_pred_feat = None):
        _, _, _, _, t = frames.size()
        feature_map = self.feature_embedding(img=frames, network=self.input_feat_gen)
        enc_in = self.pos_embedding(feature_map, t)
        enc_out = self.Encoder(enc_in)
        # queries correspond to non_pred_feat
        if non_pred_feat is not None:
            non_pred_feat = torch.concat((self.blank, non_pred_feat), dim=-1)
        else:
            non_pred_feat = self.blank

        dec_V = self.feature_embedding(img=non_pred_feat, network=self.query_feat_gen)
        dec_out = self.Decoder(dec_V, enc_out)
        out_list = []
        for i in range(prediction_count):
            out_list.append(self.back_to_pixel(dec_out[..., i]))
        x = torch.stack(out_list, dim=-1)

        return x

    def feature_embedding(self, img, network):
        generator = network
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[..., i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img
