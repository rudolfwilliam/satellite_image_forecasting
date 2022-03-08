import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model_parts.AutoencLSTM import AutoencLSTM
from .model_parts.Conv_Transformer import ENS_Conv_Transformer
from .model_parts.Conv_LSTM import Conv_LSTM
from ..losses import get_loss_from_name
from ..optimizers import get_opt_from_name
from .utils.utils import ENS

class EN_model(pl.LightningModule):
    def __init__(self, model_type, model_cfg, training_cfg):
        """
        This is the supermodel that wraps all the possible models to do satellite image forecasting.

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        
        if model_type == "ConvLSTM":
            self.model = Conv_LSTM(input_dim=self.model_cfg["input_channels"],
                                   output_dim=self.model_cfg["output_channels"],
                                   hidden_dims=self.model_cfg["hidden_channels"],
                                   big_mem=self.model_cfg["big_mem"],
                                   num_layers=self.model_cfg["n_layers"],
                                   kernel_size=self.model_cfg["kernel"],
                                   memory_kernel_size=self.model_cfg["memory_kernel"],
                                   dilation_rate=self.model_cfg["dilation_rate"],
                                   baseline=self.training_cfg["baseline"],
                                   layer_norm_flag=self.model_cfg["layer_norm"],
                                   img_width=self.model_cfg["img_width"],
                                   img_height=self.model_cfg["img_height"],
                                   peephole=self.model_cfg["peephole"])
        elif model_type == "AutoencLSTM":
            self.model = AutoencLSTM(input_dim=self.model_cfg["input_channels"],
                                     output_dim=self.model_cfg["output_channels"],
                                     hidden_dims=self.model_cfg["hidden_channels"],
                                     big_mem=self.model_cfg["big_mem"],
                                     num_layers=self.model_cfg["n_layers"],
                                     kernel_size=self.model_cfg["kernel"],
                                     memory_kernel_size=self.model_cfg["memory_kernel"],
                                     dilation_rate=self.model_cfg["dilation_rate"],
                                     baseline=self.training_cfg["baseline"],
                                     layer_norm_flag=self.model_cfg["layer_norm"],
                                     img_width=self.model_cfg["img_width"],
                                     img_height=self.model_cfg["img_height"],
                                     peephole=self.model_cfg["peephole"])
        elif model_type == "ConvTransformer":
            self.model = ENS_Conv_Transformer(num_hidden=self.model_cfg["num_hidden"],
                                              output_dim=self.model_cfg["output_channels"],
                                              depth=self.model_cfg["depth"],
                                              dilation_rate=self.model_cfg["dilation_rate"],
                                              num_conv_layers=self.model_cfg["num_conv_layers"],
                                              kernel_size=self.model_cfg["kernel_size"],
                                              img_width=self.model_cfg["img_width"],
                                              non_pred_channels=self.model_cfg["non_pred_channels"],
                                              num_layers_query_feat=self.model_cfg["num_layers_query_feat"],
                                              in_channels=self.model_cfg["in_channels"],
                                              baseline=self.training_cfg["baseline"])
        self.baseline = self.training_cfg["baseline"]
        self.future_training = self.training_cfg["future_training"]
        self.learning_rate = self.training_cfg["start_learn_rate"]
        self.training_loss = get_loss_from_name(self.training_cfg["training_loss"])
        self.test_loss = get_loss_from_name(self.training_cfg["test_loss"])

    def forward(self, x, prediction_count=1, non_pred_feat=None):
        """
        :param x: All features of the input time steps.
        :param prediction_count: The amount of time steps that should be predicted all at once.
        :param non_pred_feat: Only need if prediction_count > 1. All features that are not predicted
            by the model for all the future time steps we want to predict.
        :return: preds: Full predicted images.
        :return: predicted deltas: Predicted deltas with respect to baselines.
        :return: baselines: All future baselines as computed by the predicted deltas. Note: These are NOT the ground truth baselines.
        Do not use these for computing a loss!
        """

        preds, pred_deltas, baselines = self.model(x, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, baselines

    def batch_loss(self, batch, t_future=20, loss=None):
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
        optimizer = get_opt_from_name(self.training_cfg["optimizer"],
                                      params=self.parameters(),
                                      lr=self.training_cfg["start_learn_rate"])
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=self.training_cfg["lr_factor"], 
                                      patience=self.training_cfg["patience"],
                                      threshold=self.training_cfg["lr_threshold"],
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
        l = self.batch_loss(batch, t_future=self.future_training, loss=self.training_loss)
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
        _, l = self.batch_loss(batch, t_future=2*int(batch.size()[4]/3), loss=self.test_loss)
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
        _, l = self.batch_loss(batch, t_future=2*int(batch.size()[4]/3), loss=self.test_loss)
        return l
