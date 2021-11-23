import numpy as np
import torch.nn as nn
import torch
from .shared import Conv_Block
from collections import OrderedDict


class Conv_net(nn.Module):
    def __init__(self, input_dim, num_conv_layers, num_conv_layers_mem, hidden_dim, kernel_size, dilation_rate):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        num_conv_layers: int
            Number of convolutional blocks within the cell
        num_conv_layers_mem: int
            Number of convolutional blocks for the weight matrices that perform a hadamard product with current memory
            (should be much lower than num_conv_layers)
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.

        """

        super(Conv_LSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dilation_rate = dilation_rate
        self.num_conv_layers = num_conv_layers
        self.num_conv_layers_mem = num_conv_layers_mem
        self.kernel_size = kernel_size

        self.conv_block = Conv_Block(in_channels=self.input_dim + self.hidden_dim,
                                     out_channels=4*self.hidden_dim,
                                     dilation_rate=self.dilation_rate,
                                     num_conv_layers=self.num_conv_layers,
                                     kernel_size=self.kernel_size)
        self.conv_block_mem = Conv_Block(in_channels=self.input_dim + 2*self.hidden_dim,
                                     out_channels=3*self.hidden_dim,
                                     dilation_rate=self.dilation_rate,
                                     num_conv_layers=self.num_conv_layers_mem,
                                     kernel_size=self.kernel_size)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv_block(combined)
        combined_conv_weights = self.conv_block_mem(torch.concat([combined, c_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        w_i, w_f, w_o = torch.split(combined_conv_weights, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i + w_i * c_cur)
        f = torch.sigmoid(cc_f + w_f * c_cur)
        o = torch.sigmoid(cc_o + w_o * c_cur)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_block_mem.in_mid_conv.weight.device),  
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_block_mem.in_mid_conv.weight.device))


class Conv_LSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_conv_layers: Number of convolutional layers within the cell
        num_conv_layers_mem: Number of convolutional blocks for the weight matrices that perform a
                                 hadamard product with current memory (should be much lower than num_conv_layers)
        dilation_rate: Size of holes in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        Note: Will do same padding.
    Input:
        A tensor of shape (b, c, w, h, t)
    Output:
        The residual from the mean cube
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_conv_layers, num_conv_layers_mem,
                 num_layers, dilation_rate, batch_first=False, baseline="mean_cube"):
        super(Conv_LSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim                  # n of channels in input pics
        self.hidden_dim = hidden_dim                # n of channels that go through hidden layers
        self.kernel_size = kernel_size              # n kernel size (no magic here)
        self.num_layers = num_layers                # n of cells in time
        self.batch_first = batch_first                # true if you have c_0, h_0
        self.dilation_rate = dilation_rate
        self.num_conv_layers = num_conv_layers
        self.num_conv_layers_mem = num_conv_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(Conv_LSTM_Cell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            num_conv_layers=self.num_conv_layers,
                                            num_conv_layers_mem=self.num_conv_layers_mem,
                                            dilation_rate=self.dilation_rate))

        self.cell_list = nn.ModuleList(cell_list)
        self.baseline = baseline

    def forward(self, input_tensor, baseline, non_pred_feat=None, prediction_count=1):
        """
        Parameters
        ----------
        input_tensor:
            (b - batch_size, h - height, w - width, c - channel, t - time)
            5-D Tensor either of shape (b, c, w, h, t)
        non_pred_feat:
            non-predictive features for future frames
        baseline:
            baseline computed on the input variables. Only needed for prediction_count > 1.
        Returns
        -------
        pred_deltas
        """
        #TODO: Make code slimmer (there are some redundancies)

        b, _, w, h, _ = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        last_memory_list = []

        seq_len = input_tensor.size(-1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, :, :, :, t], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=-1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)
            last_memory_list.append(c)

        pred_deltas = [layer_output_list[-1:][0][:, :, :, :, -1]]
        baselines = [baseline]
        predictions = [torch.add(baseline, layer_output_list[-1:][0][:, :, :, :, -1])]
        
        # allow for multiple pred_deltas in a self feedback manner
        if prediction_count > 1:
            if non_pred_feat is None:
                raise ValueError('If prediction_count > 1, you need to provide non-prediction features for the '
                                 'future time steps!')
            non_pred_feat = torch.cat((torch.zeros((non_pred_feat.shape[0],
                                                    1,
                                                    non_pred_feat.shape[2],
                                                    non_pred_feat.shape[3],
                                                    non_pred_feat.shape[4]), device=non_pred_feat.device), non_pred_feat), dim = 1)

            # output from layer beneath which for the lowest layer is the prediction from the previous time step
            prev = predictions[0]
            # update the baseline & glue together predicted + given channels
            if self.baseline == "mean_cube":
                baseline = 1/(seq_len + 1) * (prev + (baseline * seq_len)) 
            else:
                baseline = prev # We don't predict image quality, so we just feed in the last prediction
            prev = torch.cat((prev, non_pred_feat[:,:,:,:,0]), axis=1)

            for counter in range(prediction_count - 1):
                for layer_idx in range(self.num_layers):
                    h, c = self.cell_list[layer_idx](input_tensor=prev, cur_state=[last_state_list[layer_idx],
                                                                                   last_memory_list[layer_idx]])
                    prev = h
                    last_state_list[layer_idx] = h
                    last_memory_list[layer_idx] = c
                    # in the last layer, make prediction
                    if layer_idx == (self.num_layers - 1):
                        pred_deltas.append(h)
                        baselines.append(baseline)
                        # next predicted entire image
                        prediction = baseline + h
                        predictions.append(prediction)
                        # update the baseline & glue together predicted + given channels
                        if self.baseline == "mean_cube":
                            baseline = 1/(seq_len + 1) * (prev + (baseline * seq_len)) 
                        else:
                            baseline = prev # We don't predict image quality, so we just feed in the last prediction
                        prev = torch.cat((prediction, non_pred_feat[:, :, :, :, counter]), axis=1)

        return predictions, pred_deltas, baselines

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param