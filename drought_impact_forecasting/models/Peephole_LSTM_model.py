from pytorch_lightning.loggers import base
import torch
import time
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import os
import glob

from torchmetrics import metric
from ..losses import cloud_mask_loss, get_loss_from_name

from .model_parts.Conv_LSTM import Peephole_Conv_LSTM
from .utils.utils import last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS
 
class Peephole_LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is roughly based on the convolutional LSTM architecture.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        c_channels = cfg["model"]["output_channels"] if cfg["model"]["big_mem"] else 1
        self.model = Peephole_Conv_LSTM(input_dim=cfg["model"]["input_channels"],
                                        output_dim=cfg["model"]["output_channels"],
                                        c_channels = c_channels,
                                        num_layers= cfg["model"]["n_layers"],
                                        kernel_size=(self.cfg["model"]["kernel"], self.cfg["model"]["kernel"]),
                                        memory_kernel_size=(self.cfg["model"]["memory_kernel"], self.cfg["model"]["memory_kernel"]),
                                        dilation_rate=self.cfg["model"]["dilation_rate"],
                                        baseline=self.cfg["model"]["baseline"])

        self.baseline = self.cfg["model"]["baseline"]
        self.val_metric = self.cfg["training"]["val_metric"]
        self.future_training = self.cfg["model"]["future_training"]
        self.learning_rate = self.cfg["training"]["start_learn_rate"]
        self.training_loss = get_loss_from_name(self.cfg["training"]["training_loss"])
        self.test_loss = get_loss_from_name(self.cfg["training"]["test_loss"])

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
        baseline = eval(self.baseline + "(x[:, 0:5, :, :, :], 4)")

        preds, pred_deltas, baselines = self.model(x, baseline=baseline, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, baselines

    def batch_loss(self, batch, t_future = 20, loss = None):
        all_data = batch
        cmc = 4 #cloud_mask channel
        T = all_data.size()[4]
        t0 = T - t_future
        context = all_data[:, :, :, :, :t0] # b, c, h, w, t
        target = all_data[:, :cmc + 1, :, :, t0:] # b, c, h, w, t
        npf = all_data[:, cmc + 1:, :, :, t0:]

        x_preds, x_delta, baselines = self(context, prediction_count=T-t0, non_pred_feat=npf)
        
        if loss is None:
            return self.training_loss(labels = target, prediction = x_preds)
        else:
            return loss(labels = target, prediction = x_preds)

    def configure_optimizers(self):
        if self.cfg["training"]["optimizer"] == "adam":
            #self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            #return self.optimizer
            optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])

            scheduler = ReduceLROnPlateau(  optimizer, 
                                            mode='min', 
                                            factor=self.cfg["training"]["lr_factor"], 
                                            patience= self.cfg["training"]["patience"],
                                            threshold=0.001,
                                            verbose=True)
        elif self.cfg["training"]["optimizer"] == "adamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])

            scheduler = ReduceLROnPlateau(  optimizer, 
                                            mode='min', 
                                            factor=self.cfg["training"]["lr_factor"], 
                                            patience= self.cfg["training"]["patience"],
                                            threshold=0.001,
                                            verbose=True)
        else:
            raise ValueError("You have specified an invalid optimizer.")
        '''
        if self.cfg["training"]["optimizer"] == "adam":
            #self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            #return self.optimizer
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])
            
            # Decay learning rate according for last (epochs - decay_point) iterations
            lambda_all = lambda epoch: self.cfg["training"]["start_learn_rate"] \
                          if epoch <= self.cfg["model"]["decay_point"] \
                          else ((self.cfg["training"]["epochs"]-epoch) / (self.cfg["training"]["epochs"]-self.cfg["model"]["decay_point"])
                                * self.cfg["training"]["start_learn_rate"])

            self.scheduler = LambdaLR(self.optimizer, lambda_all)

        else:
            raise ValueError("You have specified an invalid optimizer.")'''

        lr_sc = {
            'scheduler': scheduler,
            'monitor': 'epoch_training_loss'
        }
        return [optimizer] , [lr_sc]
    

    def training_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        l = self.batch_loss(batch, t_future=self.future_training, loss = self.training_loss)
        return l
    
    # We could try early stopping here later on
    def validation_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        _, l = self.batch_loss(batch, t_future=self.future_training, loss = self.test_loss)
        return l
    
    def test_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        _, l = self.batch_loss(batch, t_future=self.future_training, loss = self.test_loss)
        return l

