import torch
import time
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import os
import glob

from torchmetrics import metric
from ..losses import cloud_mask_loss

from .model_parts.Conv_LSTM import Conv_LSTM
from .utils.utils import last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS
 
class Conv_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is roughly based on the convolutional LSTM architecture.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]
        self.timestamp = "x"

        channels = self.cfg["model"]["channels"]
        hidden_channels = self.cfg["model"]["hidden_channels"]
        n_layers = self.cfg["model"]["n_layers"]
        self.model = Conv_LSTM(input_dim=channels,
                               dilation_rate=self.cfg["model"]["dilation_rate"],
                               hidden_dim=[hidden_channels] * n_layers,
                               kernel_size=(self.cfg["model"]["kernel"][0], self.cfg["model"]["kernel"][1]),
                               num_layers=n_layers,
                               num_conv_layers=self.cfg["model"]["num_conv_layers"],
                               num_conv_layers_mem=self.cfg["model"]["num_conv_layers_mem"],
                               batch_first=False,
                               baseline=self.cfg["model"]["baseline"])
        self.baseline = self.cfg["model"]["baseline"]
        self.val_metric = self.cfg["model"]["val_metric"]

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

        all_data = batch
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        cloud_mask_channel = 4

        T = all_data.size()[4]
        #t0 = T - 1 # no. of pics we start with
        t0 = T - 1

        _, x_delta, baseline = self(all_data[:, :, :, :, :t0])
        delta = all_data[:, :4, :, :, t0] - baseline[0]
        loss = cloud_mask_loss(x_delta[0], delta, all_data[:, cloud_mask_channel:cloud_mask_channel+1, :, :, t0])

        for t_end in range(t0 + 1, T): # this iterates with t_end = t0, ..., T-1
            _, x_delta, baseline = self(all_data[:, :, :, :, :t_end])
            delta = all_data[:, :4, :, :, t_end] - baseline[0]
            loss = loss.add(cloud_mask_loss(x_delta[0], delta, all_data[:, cloud_mask_channel:cloud_mask_channel+1, :, :, t_end]))
        
        return loss
    
    # We could try early stopping here later on
    def validation_step(self, batch, batch_idx):
        '''
            The validation step also uses the L2 loss, but on a prediction of all non-context images
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
        cloud_mask_channel = 4

        T = all_data.size()[4]
        t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

        context = all_data[:, :, :, :, :t0] # b, c, h, w, t
        target = all_data[:, :5, :, :, t0:] # b, c, h, w, t
        npf = all_data[:, 5:, :, :, t0+1:]

        x_preds, x_delta, baselines = self(context, prediction_count=T-t0, non_pred_feat=npf)
        
        if self.val_metric=="ENS":
            # ENS loss = -ENS (ENS==1 would mean perfect prediction)
            x_preds = torch.stack(x_preds , axis = -1) # b, c, h, w, t
            score, scores = ENS(prediction = x_preds, target = target)
            loss = - scores
        else: # L2 cloud mask loss
            delta = all_data[:, :4, :, :, t0] - baselines[0]
            loss = cloud_mask_loss(x_delta[0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t0])
            
            for t_end in range(t0 + 1, T): # this iterates with t_end = t0 + 1, ..., T-1
                delta = all_data[:, :4, :, :, t_end] - baselines[t_end-t0]
                loss = loss.add(cloud_mask_loss(x_delta[t_end-t0], delta, all_data[:, cloud_mask_channel:cloud_mask_channel+1, :, :, t_end]))
            
        return loss
    
    def test_step(self, batch, batch_idx):
        '''
            The test step takes the test data and makes predictions.
            They are then evaluated using the ENS score.
        '''
        #starting_time = time.time()
        all_data = batch


        T = all_data.size()[4]

        t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

        context = all_data[:, :, :, :, :t0] # b, c, h, w, t
        target = all_data[:, :5, :, :, t0:] # b, c, h, w, t
        npf = all_data[:, 5:, :, :, t0+1:]

        x_preds, x_deltas, baselines = self(x = context, 
                                            prediction_count = T-t0, 
                                            non_pred_feat = npf)
        
        x_preds = torch.stack(x_preds , axis = -1) # b, c, h, w, t
        
        score, part_scores = ENS(prediction = x_preds, target = target)
        
        # store to file the scores
        with open(os.getcwd() + "/model_instances/model_" + self.timestamp + "/scores.csv", 'a') as filehandle:
            for i in range(len(score)):
                filehandle.write(str(part_scores[i,0]) + "," +str(part_scores[i,1]) + "," + str(part_scores[i,2]) + "," + str(part_scores[i,3])+ "," + str(score[i]) + '\n')

        return part_scores

