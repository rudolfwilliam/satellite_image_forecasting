import torch.nn as nn
import torch
from .shared import Conv_Block

class Peephole_Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, h_channels, big_mem, kernel_size, memory_kernel_size, dilation_rate, layer_norm_flag, img_width, img_height):
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
        layer_norm_flag: bool
            Whether to perform layer normalization.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        """

        super(Peephole_Conv_LSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.h_channels = h_channels
        self.c_channels = h_channels if big_mem else 1
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height

        self.conv_cc = nn.Conv2d(self.input_dim + self.h_channels, self.h_channels + 3*self.c_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same', padding_mode='reflect')
        self.conv_ll = nn.Conv2d(self.c_channels, 2*self.c_channels, dilation=dilation_rate, kernel_size=memory_kernel_size,
                                     bias=False, padding='same', padding_mode='reflect')
        self.conv_oo = nn.Conv2d(self.c_channels, self.h_channels, dilation=dilation_rate, kernel_size=memory_kernel_size,
                                     bias=False, padding='same', padding_mode='reflect')
        
        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(self.input_dim + self.h_channels, affine=True)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # apply layer normalization
        if self.layer_norm_flag:
            combined = self.layer_norm(combined)
        #if self.layer_norm_flag:
        #    combined = torch.stack([self.layer_norm[c](combined[:, c, ...]) for c in range(combined.size()[1])], dim=1)

        combined_conv = self.conv_cc(combined) # h_channel + 3 * c_channel 
        combined_memory = self.conv_ll(c_cur)  # h_channel + 2 * c_channel  # NO BIAS HERE

        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, [self.c_channels, self.c_channels, self.c_channels, self.h_channels], dim=1)
        ll_i, ll_f = torch.split(combined_memory, [self.c_channels, self.c_channels], dim=1)

        i = torch.sigmoid(cc_i + ll_i)
        f = torch.sigmoid(cc_f + ll_f)

        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        oo_o = self.conv_oo(c_next) # note that o uses c_next, not c_cur
        o = torch.sigmoid(oo_o + cc_o)

        if self.h_channels == self.c_channels:
            h_next = o * torch.tanh(c_next)
        elif self.c_channels == 1:
            h_next = o * torch.tanh(c_next).repeat([1,self.h_channels, 1, 1])

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv_cc.weight.device),  
                torch.zeros(batch_size, self.c_channels, height, width, device=self.conv_cc.weight.device))

class Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, h_channels, big_mem, kernel_size, memory_kernel_size, dilation_rate, layer_norm_flag, img_width, img_height):
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
        layer_norm_flag: bool
            Whether to perform layer normalization.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        """

        super(Conv_LSTM_Cell, self).__init__()

        self.input_dim = input_dim
        self.h_channels = h_channels
        self.c_channels = h_channels if big_mem else 1
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height

        self.conv_cc = nn.Conv2d(self.input_dim + self.h_channels, self.h_channels + 3*self.c_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same', padding_mode='reflect')
        
        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(self.input_dim + self.h_channels, affine=True)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # apply layer normalization
        if self.layer_norm_flag:
            combined = self.layer_norm(combined)
        #if self.layer_norm_flag:
        #    combined = torch.stack([self.layer_norm[c](combined[:, c, ...]) for c in range(combined.size()[1])], dim=1)

        combined_conv = self.conv_cc(combined) # h_channel + 3 * c_channel 

        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, [self.c_channels, self.c_channels, self.c_channels, self.h_channels], dim=1)


        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g

        if self.h_channels == self.c_channels:
            h_next = o * torch.tanh(c_next)
        elif self.c_channels == 1:
            h_next = o * torch.tanh(c_next).repeat([1,self.h_channels, 1, 1])

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv_cc.weight.device),  
                torch.zeros(batch_size, self.c_channels, height, width, device=self.conv_cc.weight.device))

class Peephole_Conv_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, big_mem, kernel_size, memory_kernel_size, dilation_rate,
                    img_width, img_height, layer_norm_flag=True, baseline="last_frame", num_layers = 1, peephole = False):
        """
        Parameters:
            input_dim: Number of channels in input
            output_dim: Number of channels in the output
            hidden_dim: Number of channels in the hidden outputs (should be a number or a list of num_layers - 1)
            kernel_size: Size of kernel in convolutions
            memory_kernel_size: Size of kernel in convolutions when the memory influences the output
            dilation_rate: Size of holes in convolutions
            num_layers: Number of LSTM layers stacked on each other
            Note: Will do same padding.
        Input:
            A tensor of shape (b, c, w, h, t)
        Output:
            The residual from the mean cube
        """
        super(Peephole_Conv_LSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers

        self.input_dim = input_dim                                                  # n of channels in input pics
        self.h_channels = self._extend_for_multilayer(hidden_dims, num_layers - 1)  # n of hidden channels   
        self.h_channels.append(output_dim)                                          # n of channels in output pics
        self.big_mem = big_mem                                                      # true means c = h, false c = 1. 
        self.num_layers = num_layers                                                # n of channels that go through hidden layers
        self.kernel_size = kernel_size     
        self.memory_kernel_size = memory_kernel_size                                # n kernel size (no magic here)
        self.dilation_rate = dilation_rate
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.baseline = baseline
        self.peephole = peephole

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.h_channels[i - 1]
            cur_layer_norm_flag = self.layer_norm_flag if i != 0 else False
            if self.peephole:
                cell_list.append(Peephole_Conv_LSTM_Cell(input_dim=cur_input_dim,
                                                        h_channels=self.h_channels[i],
                                                        big_mem=self.big_mem,
                                                        layer_norm_flag=cur_layer_norm_flag,
                                                        img_width=self.img_width,
                                                        img_height=self.img_height,
                                                        kernel_size=self.kernel_size,
                                                        memory_kernel_size=self.memory_kernel_size,
                                                        dilation_rate=dilation_rate))
            else:
                cell_list.append(Conv_LSTM_Cell(input_dim=cur_input_dim,
                                                h_channels=self.h_channels[i],
                                                big_mem=self.big_mem,
                                                layer_norm_flag=cur_layer_norm_flag,
                                                img_width=self.img_width,
                                                img_height=self.img_height,
                                                kernel_size=self.kernel_size,
                                                memory_kernel_size=self.memory_kernel_size,
                                                dilation_rate=dilation_rate))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, baseline, non_pred_feat=None, prediction_count=1, num_layer = 1):
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

        b, _, width, height, T = input_tensor.size()
        hs = []
        cs = []

        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(b,(height,width))
            hs.append(h)
            cs.append(c)

        pred_deltas = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())
        preds = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())
        baselines = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())
        
        # iterate over the past
        for t in range(T):
            hs[0], cs[0] = self.cell_list[0](input_tensor=input_tensor[..., t], cur_state=[hs[0], cs[0]])
            for i in range(1, self.num_layers):
                hs[i], cs[i] = self.cell_list[i](input_tensor=hs[i - 1], cur_state=[hs[i], cs[i]])

        baselines[..., 0] = baseline
        pred_deltas[..., 0] = hs[-1]
        preds[..., 0] = pred_deltas[..., 0] + baselines[..., 0]
        
        # add a mask to our prediction
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

                hs[0], cs[0] = self.cell_list[0](input_tensor=prev, cur_state=[hs[0], cs[0]])
                for i in range(1, self.num_layers):
                    hs[i], cs[i] = self.cell_list[i](input_tensor=hs[i-1], cur_state=[hs[i], cs[i]])

                pred_deltas[..., t] = hs[-1]

                if self.baseline == "mean_cube":
                    baselines[..., t] = (preds[..., t-1] + (baselines[..., t - 1] * (T + t)))/(T + t + 1)
                if self.baseline == "zeros":
                    pass
                else:
                    baselines[...,t]  = preds[..., t-1]

                preds[..., t] = pred_deltas[..., t] + baselines[..., t]

        return preds, pred_deltas, baselines

    def _get_device(self):
        return self.cell_list[0].conv_cc.weight.device

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                isinstance(kernel_size, int) or
                # Lists currently not supported for Peephole_Conv_LSTM
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

        # make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim                  # n of channels in input pics
        self.hidden_dim = hidden_dim                # n of channels that go through hidden layers
        self.kernel_size = kernel_size              # n kernel size (no magic here)
        self.num_layers = num_layers                # n of cells in time
        self.batch_first = batch_first              # true if you have c_0, h_0
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
        
        pred_deltas = layer_output_list[-1:][0][:, :, :, :, -1].unsqueeze(0)
        baselines = baseline.unsqueeze(0)
        predictions = torch.add(baseline, layer_output_list[-1:][0][:, :, :, :, -1]).unsqueeze(0)

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
                seq_len += 1
            else:
                baseline = prev # no image quality predicted, just feed in last prediction
            prev = torch.cat((prev, non_pred_feat[:, :, :, :, 0]), axis=1)

            for counter in range(prediction_count - 1):
                for layer_idx in range(self.num_layers):
                    h, c = self.cell_list[layer_idx](input_tensor=prev, cur_state=[last_state_list[layer_idx],
                                                                                   last_memory_list[layer_idx]])
                    prev = h
                    last_state_list[layer_idx] = h
                    last_memory_list[layer_idx] = c
                    # in the last layer, make prediction
                    if layer_idx == (self.num_layers - 1):
                        pred_deltas = torch.cat((pred_deltas, h.unsqueeze(0)), 0)
                        baselines = torch.cat((baselines, baseline.unsqueeze(0)), 0)
                        # next predicted entire image
                        prediction = baseline + h
                        predictions = torch.cat((predictions, prediction.unsqueeze(0)), 0)
                        # update the baseline & glue together predicted + given channels
                        if self.baseline == "mean_cube":
                            baseline = (prediction + (baseline * seq_len))/(seq_len + 1)
                            seq_len += 1
                        else:
                            baseline = prediction # no image quality predicted, just feed in last prediction
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