import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np

from drought_impact_forecasting.losses import kl_weight, base_line_total_loss
from .model_parts.SAVP_model.base_model import Encoder, Discriminator_GAN, Discriminator_VAE
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

        #self.Discriminator_GAN = Discriminator_GAN(self.cfg)
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

    # For now just use the GAN
    def forward(self, x):
        return self.model(x)
        # TODO: we need to add the average since the model, for now, only predicts the delta from the average


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

        # Pls check this works correctly with pytorch lightning
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        '''
            This is not trivial: let's say we have T time steps in the cube for training. We start by taking the first t0 time samples and we try to predict the next one. We then measure the loss against the ground truth.
            Then we do the same thing by looking at t0 + 1 time samples in the dataset, to predict tbe t0 + 2. On and on until we use all but one samples to predict the last one.
        '''
        highres_dynamic, highres_static, meso_dynamic, meso_static = batch
        '''
        highres_dynamic_context and highres_dynamic_target of size (b, w, h, c, t)
            b = batch_size)
            w = width
            h = height
            c = channels
            t = time
        '''
        
        T = highres_dynamic.size()[4]
        t0 = T-1 #n of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)
        for t_end in range(t0 - 1, T - 1): # this iterate with t_end = t0, ..., T-1
            y_pred, last_state_list = self(highres_dynamic[:, :, :, :, :t_end])
            # TODO: for some reason the order in highres_dynamic seems to be b, c, w, h, t!! Not what's written in the title
            delta = highres_dynamic[:, :4, :, :, t_end + 1] - mean_cube(highres_dynamic[:, np.r_[0:4, -1:0], :, :, :], True)
            loss = loss.add(l2_crit(y_pred, delta))
        
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
        highres_dynamic, highres_static, meso_dynamic, meso_static = batch
        pass
        '''
        T = highres_dynamic.size()[4]
        t0 = T-1 #n of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)
        for t_end in range(t0 - 1, T - 1): # this iterate with t_end = t0, ..., T-1
            y_pred, last_state_list = self(highres_dynamic[:, :, :, :, :t_end])
            loss = loss.add(l2_crit(y_pred, highres_dynamic[:, :, :, :, t_end + 1]))
        
        wandb.log({"test_loss": loss})
        return loss
        '''

    def training_epoch_end(self, outputs):
        self.model()
