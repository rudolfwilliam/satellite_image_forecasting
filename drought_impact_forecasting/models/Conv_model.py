import torch
import time
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import os
import glob
from .model_parts.shared import Conv_Block

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
        out_channel = 4
        n_layers = self.cfg["model"]["n_layers"]
        kernel_size = self.cfg["model"]["kernel"]

        self.model = Conv_Block(in_channels = channels,
                                out_channels = out_channel,
                                kernel_size= kernel_size,
                                num_conv_layers= n_layers,
                                dilation_rate= self.cfg["model"]["dilation_rate"])

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
        input_data = torch.cat((baseline, x[:,4:,:,:,-1]), axis=1)
        pred = self.model(input_data)
        seq_len = 10

        pred_deltas = [pred]
        baselines = [baseline]
        predictions = [torch.add(baseline, pred)]

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
                #Only works with last!
                baseline = prev[:,:4,:,:]
                pred_delta = self.model(prev)
                next = torch.add(baseline, pred_delta)

                baselines.append(baseline)
                pred_deltas.append(pred_delta)
                predictions.append(next)

                prev = torch.cat((next, non_pred_feat[:,:,:,:,counter]), axis=1)

        return predictions, pred_deltas, baselines 

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

        non_pred_feat = all_data[:, 5:, :, :, t0:]

        _, x_delta, baseline = self(all_data[:, :, :, :, :t0], non_pred_feat = non_pred_feat, prediction_count = T-t0)
        delta = all_data[:, :4, :, :, t0:] - torch.stack(baseline, dim=-1)
        loss = cloud_mask_loss(x_delta[0], delta[:,:,:,:,0], all_data[:, cloud_mask_channel:cloud_mask_channel+1, :, :, t0])

        for i, t_end in enumerate(range(t0 + 1, T)): # this iterates with t_end = t0, ..., T-1
            loss = loss.add(cloud_mask_loss(x_delta[i+1], delta[:,:,:,:,i+1], all_data[:, cloud_mask_channel:cloud_mask_channel+1, :, :, t_end]))
            
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
        all_data = batch

        T = all_data.size()[4]

        t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

        context = all_data[:, :, :, :, :t0] # b, c, h, w, t
        target = all_data[:, :5, :, :, t0:] # b, c, h, w, t
        npf = all_data[:, 5:, :, :, t0+1:]

        x_preds, x_deltas, baselines = self(x = context, 
                                            prediction_count = T-t0, 
                                            non_pred_feat = npf)
        
        x_preds = torch.stack(x_preds, axis = -1) # b, c, h, w, t
        
        score, part_scores = ENS(prediction = x_preds, target = target)

        return part_scores

