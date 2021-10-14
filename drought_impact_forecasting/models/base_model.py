from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from drought_impact_forecasting.losses import kl_weight, base_line_total_loss
from .model_parts.base_model import Encoder, Discriminator_GAN, Discriminator_VAE

class Base_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is an adaptation of SAVP (Stochastic Adversarial Video Prediction) by Alex Lee presented in https://arxiv.org/pdf/1804.01523.pdf

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]

        self.Discriminator_GAN = Discriminator_GAN(self.cfg)

    # For now just use the GAN
    def forward(self, x):
        self.Discriminator_GAN.forward(x)

    def configure_optimizers(self):
        if self.cfg["training"]["optimizer"] == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])
            
            # Decay learning rate according for last (epochs - decay_point) iterations
            lambda_all = lambda epoch: self.cfg["training"]["start_learn_rate"] \
                          if epoch <= self.cfg["model"]["decay_point"] \
                          else ((self.cfg["training"]["epochs"]-epoch) / (30-self.cfg["model"]["decay_point"])
                                * self.cfg["training"]["start_learn_rate"])

            scheduler = LambdaLR(optimizer, lambda_all)
        else:
            raise ValueError("You have specified an invalid optimizer.")

        # Pls check this works correctly with pytorch lightning
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        y_preds = self(batch_x)
        loss_total = base_line_total_loss(y_preds, batch_y, self.current_epoch)

        return loss_total

    def validation_step(self):
        pass