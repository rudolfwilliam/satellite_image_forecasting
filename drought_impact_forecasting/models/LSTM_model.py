import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from drought_impact_forecasting.losses import kl_weight, base_line_total_loss
from .model_parts.SAVP_model.base_model import Encoder, Discriminator_GAN, Discriminator_VAE
from .model_parts.Conv_LSTM import Conv_LSTM
 
class LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is an adaptation of SAVP (Stochastic Adversarial Video Prediction) by Alex Lee presented in https://arxiv.org/pdf/1804.01523.pdf

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]

        #self.Discriminator_GAN = Discriminator_GAN(self.cfg)
        channels = 7
        n_cells = 10
        self.model = Conv_LSTM(input_dim = channels, 
                              hidden_dim = [3]*n_cells, 
                              kernel_size = (3,3), 
                              num_layers = n_cells,
                              batch_first=False, 
                              bias=True, 
                              prediction_count=1)

    # For now just use the GAN
    def forward(self, x):
        return self.model(x)

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
        '''
            This is not trivial: let's say we have T time steps in the cube for training. We start by taking the first t0 time samples and we try to predict the next one. We then measure the loss against the ground truth.
            Then we do the same thing by looking at t0 + 1 time samples in the dataset, to predict tbe t0 + 2. On and on untill we use all but one samples to predict the last one. 
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
        t0 = 2 #n of time iteration we are doing
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)
        for t_end in range(t0, T): # this iterate with t_end = t0, ..., T-1
            y_pred = self(highres_dynamic[:, :, :, :, :t_end])
            loss.add(l2_crit(y_pred, highres_dynamic[:, :, :, :, t_end + 1]))

        
        return loss
    

    """def validation_step(self):
        pass"""