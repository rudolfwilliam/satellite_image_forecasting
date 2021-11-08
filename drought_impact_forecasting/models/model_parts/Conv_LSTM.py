import numpy as np
import torch.nn as nn
import torch


class Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv_LSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class Conv_LSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        Note: Will do same padding.
    Input:
        A tensor of shape (b, c, w, h, t)
    Output:
        The residual from the mean cube
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True):
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
        self.batch_first = batch_first              # true if you have c_0, h_0
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(Conv_LSTM_Cell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, mean=None, non_pred_feat=None, hidden_state=None, prediction_count=1):
        """
        Parameters
        ----------
        input_tensor:
            (b - batch_size, h - height, w - width, c - channel, t - time)
            5-D Tensor either of shape (b, c, w, h, t)
        mean:
            mean of the input variables. Only needed for prediction_count > 1.
        Returns
        -------
        pred_deltas
        """
        b, _, w, h, _ = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

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
            last_state_list.append([h, c])

        pred_deltas = [layer_output_list[-1:][0][:, :, :, :, -1]]
        means = [mean]
        predictions = [torch.add(mean, layer_output_list[-1:][0][:, :, :, :, -1])]
        # allow for multiple pred_deltas in a self feedback manner
        if prediction_count > 1:
            if mean is None:
                raise ValueError('If prediction_count > 1, you need to provide the mean of the input images!')
            if non_pred_feat is None:
                raise ValueError('If prediction_count > 1, you need to provide non-prediction features for the '
                                 'future time steps!')
            # output from layer beneath which for the lowest layer is the prediction from the previous time step
            prev = pred_deltas[0]
            # convert to numpy array that allows for this kind of slicing
            last_state_list = np.array(last_state_list)
            last_states = last_state_list[:, 0]
            last_memories = last_state_list[:, 1]

            for counter in range(prediction_count - 1):
                for layer_idx in range(self.num_layers):
                    h, c = self.cell_list[layer_idx](input_tensor=prev, cur_state=[last_states[layer_idx],
                                                                                   last_memories[layer_idx]])
                    prev = h
                    last_states[layer_idx] = h
                    last_memories[layer_idx] = c
                    # in the last layer, make prediction
                    if layer_idx == (range(self.num_layers) - 1):
                        pred_deltas.append(h)
                        means.append(mean)
                        # next predicted entire image
                        prediction = np.sum([mean, h], axis=0)
                        predictions.append(prediction)
                        prev = np.concatenate((prediction, non_pred_feat[:, :, :, :, counter]), axis=1)
                        # update mean
                        mean = 1/(seq_len + counter + 2) * np.sum([(seq_len + counter + 1) * mean, prev], axis=0)

        return predictions, pred_deltas, means

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