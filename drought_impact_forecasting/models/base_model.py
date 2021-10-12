from torch import nn
import pytorch_lightning as pl

class Base_model(pl.LightningModule):
    def __init__(self):
        """
        Base prediction model. It is an adaptation of SAVP (Stochastic Adversarial Video Prediction) by Alex Lee presented in https://arxiv.org/pdf/1804.01523.pdf

        Args:
        """
        super().__init__()

    def forward(self):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass