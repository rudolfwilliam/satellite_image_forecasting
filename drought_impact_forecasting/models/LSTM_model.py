import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np

from .model_parts.Conv_LSTM import Conv_LSTM
from .model_parts.shared import mean_cube
 
class LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is based on the convolutional LSTM architecture.

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]

        channels = self.cfg["model"]["channels"]
        hidden_channels = self.cfg["model"]["hidden_channels"]
        n_layers = self.cfg["model"]["n_layers"]
        self.model = Conv_LSTM(input_dim=channels,
                              hidden_dim=[hidden_channels] * n_layers,
                              kernel_size=(self.cfg["model"]["kernel"][0], self.cfg["model"]["kernel"][1]),
                              num_layers=n_layers,
                              batch_first=False, 
                              bias=True, 
                              prediction_count=1)

    def forward(self, x):
        # Compute mean cube
        mean = mean_cube(x[:, np.r_[0:5], :, :, :], True)
        pred_delta = self.model(x)
        # Prediction is mean + residual
        pred = pred_delta + mean
        return pred, pred_delta, mean

    def configure_optimizers(self):
        if self.cfg["training"]["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])
            
            # Decay learning rate according for last (epochs - decay_point) iterations
            lambda_all = lambda epoch: self.cfg["training"]["start_learn_rate"] \
                          if epoch <= self.cfg["model"]["decay_point"] \
                          else ((self.cfg["training"]["epochs"]-epoch) / (self.cfg["training"]["epochs"]-self.cfg["model"]["decay_point"])
                                * self.cfg["training"]["start_learn_rate"])

            self.scheduler = LambdaLR(self.optimizer, lambda_all)
        else:
            raise ValueError("You have specified an invalid optimizer.")

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        '''
            This is not trivial: let's say we have T time steps in the cube for training.
            We start by taking the first t0 time samples and we try to predict the next one.
            We then measure the loss against the ground truth.
            Then we do the same thing by looking at t0 + 1 time samples in the dataset, to predict the t0 + 2.
            On and on until we use all but one samples to predict the last one.
        '''
        all_data = batch
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''

        T = all_data.size()[4]
        t0 = T - 20 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)   ########## CHECK USE OF REQUIRES_GRAD
        for t_end in range(t0, T-1): # this iterates with t_end = t0, ..., T-1
            x_pred, x_delta, mean = self(all_data[:, :, :, :, :t_end])
            delta = all_data[:, :4, :, :, t_end + 1] - mean
            loss = loss.add(l2_crit(x_delta, delta))
        
        logs = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    # We could try early stopping here later on
    """def validation_step(self):
        pass"""

    def test_step(self, batch, batch_idx):
        '''
            TBD: Here we could directly incorporate the EarthNet Score from the model demo.
        '''
        all_data = batch
        context = all_data # here we will store the context + until now predicted images

        T = all_data.size()[4]
        t0 = 10 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)
        for t_end in range(t0 - 1, T - 1): # this iterates with t_end = t0, ..., T-1
            x_pred, x_delta, mean = self(context[:, :, :, :, :t_end]) # why x_pred, not y_pred
            context[:,:,:,:,t_end] = x_pred
            delta = all_data[:, :4, :, :, t_end + 1] - mean
            loss = loss.add(l2_crit(x_delta, delta))
        
        logs = {'test_loss': loss}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
