import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import os
import glob
from ..losses import cloud_mask_loss
from .model_parts.Conv_Transformer import Conv_Transformer
from .model_parts.shared import last_cube, mean_cube, mean_prediction, last_prediction, get_ENS
import pytorch_lightning as pl


class Transformer_model(pl.LightningModule):

    def __init__(self, cfg, timestamp):
        """
        State of the art prediction model. It is roughly based on the ConvTransformer architecture.
        (https://arxiv.org/pdf/2011.10185.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]
        self.timestamp = timestamp


        self.model = Conv_Transformer(configs=self.cfg["model"])

    def forward(self, x, prediction_count=1, non_pred_feat=None):
        """
        :param x: All features of the input time steps.
        :param prediction_count: The amount of time steps that should be predicted all at once.
        :param non_pred_feat: Only need if prediction_count > 1. All features that are not predicted
        by the model for all the future to be predicted time steps.
        :return: preds: Full predicted images.
        :return: predicted deltas: Predicted deltas with respect to means.
        :return: means: All future means as computed by the predicted deltas. Note: These are NOT the ground truth means.
        Do not use these for computing a loss!
        """
        # compute mean cube
        mean = mean_cube(x[:, 0:5, :, :, :], 4)
        preds, pred_deltas, means = self.model(x, mean=mean, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, means

