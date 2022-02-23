from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .shared import Conv_Block
from ..utils.utils import zeros, last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS



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
    def __init__(self, kernel_size, num_hidden, dilation_rate, num_conv_layers):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_hidden = num_hidden
        self.num_conv_layers = num_conv_layers
        self.dilation_rate = dilation_rate
        self.conv = Conv_Block(self.num_hidden, self.num_hidden, kernel_size=self.kernel_size,
                               dilation_rate=self.dilation_rate, num_conv_layers=self.num_conv_layers)

    def forward(self, x):
        return torch.stack([self.conv(x[..., i]) for i in range(x.size()[-1])], dim=-1)


class ConvAttention(nn.Module):
    def __init__(self, num_hidden, kernel_size, enc=True, mask=False):
        super(ConvAttention, self).__init__()
        self.enc = enc
        self.mask = mask
        self.kernel_size = kernel_size
        self.num_hidden = num_hidden
        # important note: shared convolution is intentional here
        if self.enc:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=3*self.num_hidden, kernel_size=1, padding="same", padding_mode="reflect")
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.num_hidden, out_channels=2*self.num_hidden, kernel_size=1, padding="same", padding_mode="reflect")
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden*2, out_channels=1, kernel_size=self.kernel_size, padding="same", padding_mode="reflect")
        )

    def forward(self, x, enc_out=None):
        b, _, _, _, s = x.shape
        if self.enc:
            qkv_setlist = []
            t = s
            for i in range(t):
                qkv_setlist.append(self.conv1(x[..., i]))
            qkv_set = torch.stack(qkv_setlist, dim=-1)
            Q, K, V = torch.split(qkv_set, self.num_hidden, dim=1)
        else:
            # x corresponds to the query
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
                Q_K_temp = rearrange(Q_K_temp, 'b c h w t -> (b t) c h w') # no convolution across time dim!
                extr_feat = rearrange(torch.squeeze(self.conv2(Q_K_temp), dim=1), '(b t) h w -> b h w t', b=b, t=i+1)
                attn_mask = F.softmax(extr_feat, dim=-1)
                V_pre = torch.stack([torch.mul(attn_mask, V_rep[:, c, :, :, i, :i+1]) for c in range(V_rep.size()[1])], dim=1)
                V_out.append(torch.sum(V_pre, dim=-1))
            V_out = torch.stack(V_out, dim=-1)
        else:
            Q_K = rearrange(Q_K, 'b c h w s t -> (b s t) c h w') # no convolution across time dim!
            extr_feat = rearrange(torch.squeeze(self.conv2(Q_K), dim=1), '(b s t) h w -> b h w t s', b=b, t=t)
            attn_mask = F.softmax(extr_feat, dim=-2)
            V_pre = torch.stack([torch.mul(attn_mask, V_rep[:, c, ...]) for c in range(V_rep.size()[1])], dim=1)
            V_out = torch.sum(V_pre, dim=-2)

        return V_out


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, img_width):
        super(PositionalEncoding, self).__init__()
        self.num_hidden = num_hidden
        self.img_width = img_width

    def _get_sinusoid_encoding_table(self, t, device):
        ''' Sinusoid position encoding table '''
        # no differentiation should happen with respect to the params in here!

        def get_position_angle_vec(position):
            return_list = [torch.ones((1,
                                       self.img_width,
                                       self.img_width),
                                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) * 
                                       (position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden[-1])) for hid_j in range(self.num_hidden[-1])]
            return torch.stack(return_list, dim=1)

        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(t)]
        sinusoid_table = torch.stack(sinusoid_table, dim=0)
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # even dim
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # odd dim

        return torch.moveaxis(sinusoid_table, 0, -1)

    def forward(self, x, t):
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(t, x.get_device()))

        return torch.squeeze(x + self.pos_table.clone().detach(), dim=0)


class Encoder(nn.Module):
    def __init__(self, num_hidden, depth, dilation_rate, num_conv_layers, kernel_size, img_width):
        super().__init__()
        self.num_hidden = num_hidden
        self.depth = depth
        self.dilation_rate = dilation_rate
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.img_width = img_width
        self.layers = nn.ModuleList([])
        self.num_hidden = self.num_hidden
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.img_width, self.img_width],
                                 ConvAttention(kernel_size=self.kernel_size, num_hidden=self.num_hidden[-1], enc=True))),
                Residual(PreNorm([self.num_hidden[-1], self.img_width, self.img_width],
                                 FeedForward(kernel_size=self.kernel_size, num_hidden=self.num_hidden[-1], 
                                 dilation_rate=self.dilation_rate, num_conv_layers=self.num_conv_layers)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_hidden, depth, dilation_rate, num_conv_layers, kernel_size, img_width, non_pred_channels):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dilation_rate = dilation_rate
        self.num_conv_layers = num_conv_layers
        self.depth = depth
        self.kernel_size = kernel_size
        self.img_width = img_width
        self.num_hidden = num_hidden
        self.num_non_pred_feat = non_pred_channels
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                # (masked) query self-attention
                Residual(PreNorm([self.num_hidden[-1], self.img_width, self.img_width],
                                     ConvAttention(num_hidden=self.num_hidden[-1], kernel_size=self.kernel_size, mask=True))),
                # encoder-decoder attention
                Residual(PreNorm([self.num_hidden[-1], self.img_width, self.img_width],
                                 ConvAttention(num_hidden=self.num_hidden[-1], kernel_size=self.kernel_size, enc=False))),
                # feed forward
                Residual(PreNorm([self.num_hidden[-1], self.img_width, self.img_width],
                                 FeedForward(num_hidden=self.num_hidden[-1], kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, num_conv_layers=self.num_conv_layers)))
            ]))

    def forward(self, queries, enc_out):
        for query_attn, attn, ff in self.layers:
            queries = query_attn(queries)
            x = attn(queries, enc_out=enc_out)
            x = ff(x)

        return x

class Conv_Transformer(nn.Module):

    """Standard, single-headed ConvTransformer like in https://arxiv.org/pdf/2011.10185.pdf"""

    def __init__(self, num_hidden, depth, dilation_rate, num_conv_layers, kernel_size, img_width, non_pred_channels, num_layers_query_feat, in_channels):
        super(Conv_Transformer, self).__init__()
        self.num_hidden = num_hidden
        self.depth = depth
        self.num_layers_query_feat = num_layers_query_feat
        self.dilation_rate = dilation_rate
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.img_width = img_width
        self.in_channels = in_channels
        self.non_pred_channels = non_pred_channels
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.img_width)
        self.Encoder = Encoder(num_hidden=self.num_hidden, depth=self.depth, dilation_rate=self.dilation_rate, 
                               num_conv_layers=self.num_conv_layers, kernel_size=self.kernel_size, img_width=self.img_width)
        self.Decoder = Decoder(num_hidden=self.num_hidden, depth=self.depth, dilation_rate=self.dilation_rate, 
                               num_conv_layers=self.num_conv_layers, kernel_size=self.kernel_size, img_width=self.img_width, non_pred_channels=self.non_pred_channels)
        self.input_feat_gen = Conv_Block(self.in_channels, self.num_hidden[-1], num_conv_layers=self.num_conv_layers, kernel_size=self.kernel_size)
        # TODO (optionally): replace this by SFFN
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], 4, kernel_size=1)
        )

    def forward(self, frames, n_predictions):
        _, _, _, _, T = frames.size()
        feature_map = self.feature_embedding(img=frames, network=self.input_feat_gen)
        enc_in = self.pos_embedding(feature_map, T)
        # encode all input values
        enc_out = torch.concat(self.Encoder(enc_in), dim=-1)

        out_list = []
        queries = self.feature_embedding(img=feature_map[..., -1], network=self.query_feat_gen)
        for _ in range(n_predictions):
            dec_out = self.Decoder(queries, enc_out)
            pred = self.feature_embedding(dec_out)
            out_list.append(pred)
            queries = torch.concat((queries, pred), dim=-1)
        
        x = torch.stack(out_list, dim=-1)

        return x

    def feature_embedding(self, img, network):
        generator = network
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[..., i]))
        gen_img = torch.stack(gen_img, dim=-1)

        return gen_img

class ENS_Conv_Transformer(Conv_Transformer):

    """ConvTransformer that employs delta model and can read in non-pred future features, hence taylored to the ENS challenge."""

    def __init__(self, num_hidden, output_dim, depth, dilation_rate, num_conv_layers, kernel_size, img_width, non_pred_channels, num_layers_query_feat, in_channels, baseline):
        super(ENS_Conv_Transformer, self).__init__(num_hidden, depth, dilation_rate, num_conv_layers, kernel_size, img_width, non_pred_channels, num_layers_query_feat, in_channels - 1)
        # remove cloud mask
        self.in_channels = self.in_channels - 1
        self.baseline = baseline
        self.output_dim = output_dim
    
    def forward(self, input_tensor, non_pred_feat=None, prediction_count=1):
        baseline = eval(self.baseline + "(input_tensor[:, 0:5, :, :, :], 4)")

        b, _, width, height, T = input_tensor.size()

        pred_deltas = torch.zeros((b, self.output_dim, height, width, prediction_count), device = self._get_device())
        preds = torch.zeros((b, self.output_dim, height, width, prediction_count), device = self._get_device())
        baselines = torch.zeros((b, self.output_dim, height, width, prediction_count), device = self._get_device())

        # remove cloud mask channel for feature embedding
        feature_map = torch.concat((input_tensor[:, :4, ...], input_tensor[:, 5:, ...]), dim=1)
        features = self.feature_embedding(img=feature_map, network=self.input_feat_gen)
         
        enc_in = torch.stack([self.pos_embedding(features[i, ...], T) for i in range(b)], dim=0)
        enc_out = self.Encoder(enc_in)

        # first query stems from last input frame
        queries = features[..., -1:]
        baselines[..., 0] = baseline
        pred_deltas[..., 0] = self.back_to_pixel(self.Decoder(queries, enc_out)[..., 0])
        preds[..., 0] = pred_deltas[..., 0] + baselines[..., 0]

        for t in range(1, prediction_count):
            if self.baseline == "mean_cube":
                baselines[..., t] = (preds[..., t - 1] + (baselines[..., t - 1] * (T + t)))/(T + t + 1)
            if self.baseline == "zeros":
                pass
            else:
                baselines[..., t]  = preds[..., t - 1]

            # concatenate with non-pred features & feature embedding
            queries = torch.concat((queries, self.feature_embedding(torch.concat((preds[..., t-1:t], non_pred_feat[..., t-1:t]), dim=1), network=self.input_feat_gen)), dim=-1)
            pred_deltas[..., :t] = torch.stack([self.back_to_pixel(self.Decoder(queries, enc_out)[..., i]) for i in range(t)], dim=-1)

            preds[..., t] = pred_deltas[..., t] + baselines[..., t]

        return preds, pred_deltas, baselines
    
    def _get_device(self):
        return next(self.parameters()).device 