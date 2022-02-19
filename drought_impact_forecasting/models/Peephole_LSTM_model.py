import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..losses import get_loss_from_name
from ..optimizers import get_opt_from_name
from .model_parts.Peeph_Conv_LSTM import Peephole_Conv_LSTM
from .utils.utils import zeros, last_cube, mean_cube, last_frame, mean_prediction, last_prediction, get_ENS, ENS

class Peephole_LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        This architecture is roughly based on the convolutional LSTM architecture, with an additional peephole.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = Peephole_Conv_LSTM(input_dim=cfg["model"]["input_channels"],
                                        output_dim=cfg["model"]["output_channels"],
                                        hidden_dims=cfg["model"]["hidden_channels"],
                                        big_mem=cfg["model"]["big_mem"],
                                        num_layers=cfg["model"]["n_layers"],
                                        kernel_size=self.cfg["model"]["kernel"],
                                        memory_kernel_size=self.cfg["model"]["memory_kernel"],
                                        dilation_rate=self.cfg["model"]["dilation_rate"],
                                        baseline=self.cfg["model"]["baseline"],
                                        layer_norm_flag=cfg["model"]["layer_norm"],
                                        img_width=cfg["model"]["img_width"],
                                        img_height=cfg["model"]["img_height"])

        self.baseline = self.cfg["model"]["baseline"]
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
        cmc = 4 # cloud_mask channel
        T = batch.size()[4]
        t0 = T - t_future
        context = batch[:, :, :, :, :t0]       # b, c, h, w, t
        target = batch[:, :cmc + 1, :, :, t0:] # b, c, h, w, t
        npf = batch[:, cmc + 1:, :, :, t0:]

        x_preds, _, _ = self(context, prediction_count=T-t0, non_pred_feat=npf)
        
        if loss is None:
            return self.training_loss(labels=target, prediction=x_preds)
        else:
            return loss(labels=target, prediction=x_preds)

    def configure_optimizers(self):        
        optimizer = get_opt_from_name(self.cfg["training"]["optimizer"],
                                      params=self.parameters(),
                                      lr=self.cfg["training"]["start_learn_rate"])
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=self.cfg["training"]["lr_factor"], 
                                      patience= self.cfg["training"]["patience"],
                                      threshold=self.cfg["training"]["lr_threshold"],
                                      verbose=True)
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
    
    def validation_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        _, l = self.batch_loss(batch, t_future=2*int(batch.size()[4]/3), loss = self.test_loss)
        v_loss = np.mean(np.vstack(l), axis = 0)
        if np.min(v_loss[1:]) == 0:
            v_loss[0] = 0
        else:
            v_loss[0] = 4 / (1 / v_loss[1] + 1 / v_loss[2] + 1 / v_loss[3] + 1 / v_loss[4])
        self.log('epoch_validation_ENS', v_loss[0], on_epoch=True, on_step=False)
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
        # across all test sets 1/3 is context, 2/3 target
        _, l = self.batch_loss(batch, t_future=2*int(batch.size()[4]/3), loss = self.test_loss)
        return l
