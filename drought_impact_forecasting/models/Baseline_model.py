import torch
import numpy as np
import pytorch_lightning as pl
from .utils.utils import last_frame, ENS

class Last_model(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, x, prediction_count = 1):
        # compute the baseline
        preds = last_frame(x)
        preds = preds.unsqueeze(-1).repeat(1, 1, 1, 1, prediction_count)
        return preds

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):

        '''
            The test step takes the test data and makes predictions.
            They are then evaluated using the ENS score.
        '''
        all_data, _ = batch


        T = all_data.size()[4]

        t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

        context = all_data[:, :5 , :, :, :t0] # b, c, h, w, t
        target  = all_data[:, :5, :, :, t0:] # b, c, h, w, t
        npf     = all_data[:, 5:, :, :, t0+1:]

        preds = last_frame(context)
        x_preds = self(x = context, prediction_count = T - t0)
        
        # b, c, h, w, t
        
        score, _ = ENS(prediction = x_preds, target = target)
        
        logs = {'y_loss': np.mean(score)}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # store to file the scores
        return logs