import torch
import time
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import os
import glob
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from pytorch_lightning.loggers import base
from torchmetrics import metric
from ..losses import cloud_mask_loss
from ..losses import get_loss_from_name
from .model_parts.Conv_LSTM import Conv_LSTM
from .utils.utils import last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS
 
class NDVI_Peephole_LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is roughly based on the convolutional LSTM architecture.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        c_channels = 1
        self.model = Peephole_Conv_LSTM(input_dim=cfg["model"]["input_channels"],
                                        output_dim=1,
                                        c_channels=c_channels,
                                        num_layers=cfg["model"]["n_layers"],
                                        kernel_size=(self.cfg["model"]["kernel"], self.cfg["model"]["kernel"]),
                                        memory_kernel_size=(self.cfg["model"]["memory_kernel"], self.cfg["model"]["memory_kernel"]),
                                        dilation_rate=self.cfg["model"]["dilation_rate"],
                                        baseline=self.cfg["model"]["baseline"])
        self.training_loss = get_loss_from_name(self.cfg["model"]["training_loss"])
        self.test_loss = get_loss_from_name(self.cfg["model"]["test_loss"])
        self.baseline = self.cfg["model"]["baseline"]
        self.val_metric = self.cfg["training"]["val_metric"]
        self.future_training = self.cfg["model"]["future_training"]
        self.learning_rate = self.cfg["training"]["start_learn_rate"]

    def forward(self, x, prediction_count=1, non_pred_feat=None):
        """
        :param x: All features of the input time steps.
        :param prediction_count: The amount of time steps that should be predicted all at once.
        :param non_pred_feat: Only need if prediction_count > 1. All features that are not predicted
        by the model for all the future to be predicted time steps.
        :return: preds: Full predicted images.
        :return: predicted deltas: Predicted deltas with respect to baselines.
        :return: baselines: All future baselines as computed by the predicted deltas. Note: These are NOT the ground truth baselines.
        Do not use these for computing a loss!
        """
        # compute the baseline
        baseline = eval(self.baseline + "(x[:, 0:2, :, :, :], 1)")

        preds, pred_deltas, baselines = self.model(x, baseline=baseline, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, baselines

    def configure_optimizers(self):
        if self.cfg["training"]["optimizer"] == "adam":
            #self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            #return self.optimizer
            optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])

            scheduler = ReduceLROnPlateau(optimizer, 
                                          mode='min', 
                                          factor=self.cfg["training"]["lr_factor"], 
                                          patience= self.cfg["training"]["patience"],
                                          threshold=0.001,
                                          verbose=True)

        else:
            raise ValueError("You have specified an invalid optimizer.")

        lr_sc = {
            'scheduler': scheduler,
            'monitor': 'train loss_epoch'
        }
        return [optimizer] , [lr_sc]
        
    def batch_loss(self, batch, t_future = 20, loss = None):
        all_data = batch
        cloud_mask_channel = 2
        T = all_data.size()[4]
        t0 = T - t_future
        context = all_data[:, :, :, :, :t0] # b, c, h, w, t
        target = all_data[:, :2, :, :, t0:] # b, c, h, w, t
        npf = all_data[:, 2:, :, :, t0:]

        x_preds, _, _ = self(context, prediction_count=T-t0, non_pred_feat=npf)
        
        if loss is None:
            l = self.training_loss(x_preds, target)
        else:
            l = loss(x_preds, target)
        return l

    def training_step(self, batch, batch_idx):
        l = self.batch_loss(batch, t_future = self.future_training, loss = self.training_loss)
        self.log("train loss", l, on_step= True, on_epoch= True)
        return l
    
    def validation_step(self, batch, batch_idx):
        l = self.batch_loss(batch, t_future = self.future_training, loss = self.test_loss)
        self.log("val loss", l, on_step= False, on_epoch= True)
        return l
    
    def test_step(self, batch, batch_idx):
        l = self.batch_loss(batch, t_future = self.future_training, loss = self.test_loss)
        self.log("test loss", l, on_step= False, on_epoch= True)
        return l

