import torch.nn as nn
import torch
from ..utils.utils import zeros, mean_cube, last_frame, ENS
from .Conv_LSTM import Conv_LSTM_Cell

class AutoencLSTM(nn.Module):
    """Encoder-Decoder architecture based on ConvLSTM"""
    def __init__(self, input_dim, output_dim, hidden_dims, big_mem, kernel_size, memory_kernel_size, dilation_rate,
                    img_width, img_height, layer_norm_flag=False, baseline="last_frame", num_layers=1, peephole=True):
        super(AutoencLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers

        self.input_dim = input_dim
        self.h_channels = [[], []]                                                     # n of channels in input pics
        self.h_channels[0] = self._extend_for_multilayer(hidden_dims, num_layers)      # n of hidden channels for encoder cells
        self.h_channels[1] = self._extend_for_multilayer(hidden_dims, num_layers - 1)  # n of hidden channels for decoder cells 
        self.h_channels[1].append(output_dim)                                          # n of channels in output pics
        self.big_mem = big_mem                                                         # true means c = h, false c = 1. 
        self.num_layers = num_layers                                                   # n of channels that go through hidden layers
        self.kernel_size = kernel_size     
        self.memory_kernel_size = memory_kernel_size                                   # n kernel size (no magic here)
        self.dilation_rate = dilation_rate
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.baseline = baseline
        self.peephole = peephole

        cur_input_dim = [self.input_dim if i == 0 else self.h_channels[0][i - 1] for i in range(self.num_layers)]
        self.ENC = nn.ModuleList([Conv_LSTM_Cell(cur_input_dim[i], self.h_channels[0][i], big_mem, kernel_size, memory_kernel_size, dilation_rate, 
                                                 layer_norm_flag, img_width, img_height, peephole) for i in range(num_layers)])
        self.DEC = nn.ModuleList([Conv_LSTM_Cell(self.h_channels[0][i], self.h_channels[1][i], big_mem, kernel_size, memory_kernel_size, dilation_rate, 
                                                 layer_norm_flag, img_width, img_height, peephole) for i in range(num_layers)])

    def forward(self, input_tensor, non_pred_feat=None, prediction_count=1):
        baseline = eval(self.baseline + "(input_tensor[:, 0:5, :, :, :], 4)")
        b, _, width, height, T = input_tensor.size()
        hs = [[], []]
        cs = [[], []]

        # For encoder and decoder
        for j, part in enumerate([self.ENC, self.DEC]):
            for i in range(self.num_layers):
                h, c = part[i].init_hidden(b, (height, width))
                hs[j].append(h)
                cs[j].append(c)

        pred_deltas = torch.zeros((b, self.h_channels[1][-1], height, width, prediction_count), device = self._get_device())
        preds = torch.zeros((b, self.h_channels[1][-1], height, width, prediction_count), device = self._get_device())
        baselines = torch.zeros((b, self.h_channels[1][-1], height, width, prediction_count), device = self._get_device())

        # iterate over the past
        for t in range(T):
            hs[0][0], cs[0][0] = self.ENC[0](input_tensor=input_tensor[..., t], cur_state=[hs[0][0], cs[0][0]])
            hs[1][0], cs[1][0] = self.DEC[0](input_tensor=hs[0][0], cur_state=[hs[1][0], cs[1][0]])
            for i in range(1, self.num_layers):
                # encode
                hs[0][i], cs[0][i] = self.ENC[i](input_tensor=hs[0][i - 1], cur_state=[hs[0][i], cs[0][i]])
                # decode
                hs[1][i], cs[1][i] = self.DEC[i](input_tensor=hs[0][i], cur_state=[hs[1][i], cs[1][i]])
        
        baselines[..., 0] = baseline
        pred_deltas[..., 0] = hs[1][-1]
        preds[..., 0] = pred_deltas[..., 0] + baselines[..., 0]

        # add a mask to prediction
        if prediction_count > 1:
            non_pred_feat = torch.cat((torch.zeros((non_pred_feat.shape[0],
                                                    1,
                                                    non_pred_feat.shape[2],
                                                    non_pred_feat.shape[3],
                                                    non_pred_feat.shape[4]), device=non_pred_feat.device), non_pred_feat), dim = 1)

            # iterate over the future
            for t in range(1, prediction_count):
                # glue together with non_pred_data
                prev = torch.cat((preds[..., t - 1], non_pred_feat[..., t - 1]), axis=1)

                hs[0][0], cs[0][0] = self.ENC[0](input_tensor=prev, cur_state=[hs[0][0], cs[0][0]])
                hs[1][0], cs[1][0] = self.DEC[0](input_tensor=hs[0][0], cur_state=[hs[1][0], cs[1][0]])
                for i in range(1, self.num_layers):
                    # encode
                    hs[0][i], cs[0][i] = self.ENC[i](input_tensor=hs[0][i - 1], cur_state=[hs[0][i], cs[0][i]])
                    # decode
                    hs[1][i], cs[1][i] = self.DEC[i](input_tensor=hs[0][i], cur_state=[hs[1][i], cs[1][i]]) 

                pred_deltas[..., t] = hs[1][-1]

                if self.baseline == "mean_cube":
                    baselines[..., t] = (preds[..., t-1] + (baselines[..., t-1] * (T + t)))/(T + t + 1)
                if self.baseline == "zeros":
                    pass
                else:
                    baselines[..., t]  = preds[..., t-1]

                preds[..., t] = pred_deltas[..., t] + baselines[..., t]

        return preds, pred_deltas, baselines
    
    def _get_device(self):
        return next(self.parameters()).device

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                isinstance(kernel_size, int) or
                # lists are currently not supported for Peephole_Conv_LSTM
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, rep):
        if not isinstance(param, list):
            if rep > 0:
                param = [param] * rep
            else:
                return []
        return param

            



