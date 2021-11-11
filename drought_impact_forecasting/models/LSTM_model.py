import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import os
import glob

from .model_parts.Conv_LSTM import Conv_LSTM
from .model_parts.shared import mean_cube, mean_prediction, last_prediction, get_ENS
 
class LSTM_model(pl.LightningModule):
    def __init__(self, cfg):
        """
        Base prediction model. It is based on the convolutional LSTM architecture.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]

        channels = self.cfg["model"]["channels"]
        hidden_channels = self.cfg["model"]["hidden_channels"]
        n_layers = self.cfg["model"]["n_layers"]
        self.model = Conv_LSTM(input_dim=channels,
                              hidden_dim=[hidden_channels] * n_layers,
                              kernel_size=(self.cfg["model"]["kernel"][0], self.cfg["model"]["kernel"][1]),
                              num_layers=n_layers,
                              batch_first=False, 
                              bias=True)

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
        mean = mean_cube(x[:, 0:5, :, :, :], True)
        preds, pred_deltas, means = self.model(x, mean=mean, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, means

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
        '''
            This is not trivial: let's say we have T time steps in the cube for training.
            We start by taking the first t0 time samples and we try to predict the next one.
            We then measure the loss against the ground truth.
            Then we do the same thing by looking at t0 + 1 time samples in the dataset, to predict the t0 + 2.
            On and on until we use all but one samples to predict the last one.
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

        T = all_data.size()[4]
        t0 = T - 20 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0], requires_grad = True)   ########## CHECK USE OF REQUIRES_GRAD
        for t_end in range(t0, T): # this iterates with t_end = t0, ..., T-1
            x_pr, x_delta, mean = self(all_data[:, :, :, :, :t_end])
            delta = all_data[:, :4, :, :, t_end] - mean[0]
            loss = loss.add(l2_crit(x_delta[0], delta))
        
        logs = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    # We could try early stopping here later on
    """def validation_step(self):
        pass"""

    def test_step(self, batch, batch_idx):
        '''
            TODO: Here we could directly incorporate the EarthNet Score from the model demo.
        '''
        all_data = batch

        T = all_data.size()[4]
        t0 = 10 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0])

        x_preds, x_deltas, means = self(all_data[:, :, :, :, :t0], prediction_count=T-t0, non_pred_feat=all_data[:,4:,:,:,t0+1:])

        # Add up losses across all timesteps
        for i in range(len(means)):
            delta = all_data[:,:4,:,:,t0+i] - means[i]
            loss = loss.add(l2_crit(x_deltas[i], delta))
        
        logs = {'test_loss': loss}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        x_preds = np.array(torch.cat(x_preds, axis=0)).transpose(2,3,1,0)
        
        # Make all our predictions and save them
        if self.cfg["project"]["evaluate"]:
            # Store predictions ready for evaluation
            pred_dir = os.getcwd() + '/Data/predictions/' + str(batch_idx) + '/'
            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)

            num_context = round(all_data.shape[-1]/3)
            # Save avg predictions
            avg_cube = mean_prediction(all_data[:, 0:5, :, :, :num_context], True, num_context*2)
            np.savez(pred_dir+'pred1', avg_cube)
            # Save last cloud-free image predictions
            last_cube = last_prediction(all_data[:, 0:5, :, :, :num_context], True, num_context*2)
            np.savez(pred_dir+'pred2', last_cube)
            # Save our model prediction
            np.savez(pred_dir+'pred3', x_preds)

            predictions = [pred_dir+'pred1.npz', pred_dir+'pred2.npz', pred_dir+'pred3.npz']
            # Calculate ENS scores
            target_files = []
            with open(os.getcwd() + self.cfg["data"]["test_dir"] + '/target_files.txt', 'r') as filehandle:
                for line in filehandle:
                    # remove linebreak which is the last character of the string
                    cur = line[:-1]
                    target_files.append(cur)
            target_file = target_files[batch_idx]

            scores = get_ENS(target_file, predictions)
            best_score = max(scores)

            files = glob.glob(os.getcwd() + '/Data/scores/*.txt')
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'a') as filehandle:
                filehandle.write('Batch ' + str(batch_idx) + ' scores: ' + str(scores) + ' Best: ' + str(best_score) + '\n')