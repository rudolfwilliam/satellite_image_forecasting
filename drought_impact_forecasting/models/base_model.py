from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
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

    def forward(self):
        pass

    def configure_optimizers(self):
        if self.cfg["training"]["optimizer"] == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg["training"]["start_learn_rate"])
        else:
            raise ValueError("You have specified an invalid optimizer.")

        return [optimizer]

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        # Useless code?
        """
        L1_criterion = nn.L1Loss()
        KL_criterion = nn.KLDivLoss()
        GAN_criterion = nn.CrossEntropyLoss() 
        """


        # TODO: Adapt adaptive learning rate to new code structure/find a pytorch technique for doing this

        """curLearningRate = self.__get_learning_rate(i)
        for g in optimizer.param_groups:
            g['lr'] = curLearningRate"""

        y_preds = self(batch_x)
        loss_total = base_line_total_loss(y_preds, batch_y, self.current_epoch)

        return loss_total

    def validation_step(self):
        pass

    """def __get_learning_rate(self, epoch):
        if epoch < decayPoint:
            return startLearnRate
        else:
            startLearnRate * ((epoch - decayPoint) / (epochs - decayPoint))"""