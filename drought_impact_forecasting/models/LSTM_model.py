import torch
import time
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import os
import glob
from ..losses import cloud_mask_loss

from .model_parts.Conv_LSTM import Conv_LSTM
from .model_parts.shared import last_cube, mean_cube, mean_prediction, last_prediction, get_ENS
 
class LSTM_model(pl.LightningModule):
    def __init__(self, cfg, timestamp):
        """
        Base prediction model. It is based on the convolutional LSTM architecture.
        (https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.cfg = cfg
        self.num_epochs = self.cfg["training"]["epochs"]
        self.timestamp = timestamp

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
        mean = mean_cube(x[:, 0:5, :, :, :], 4)
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
        all_data, _ = batch
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
        t0 = T - 1 # no. of pics we start with
        loss = torch.tensor([0.0], requires_grad = True)   ########## CHECK USE OF REQUIRES_GRAD
        for t_end in range(t0, T): # this iterates with t_end = t0, ..., T-1
            x_pr, x_delta, mean = self(all_data[:, :, :, :, :t_end])
            delta = all_data[:, :4, :, :, t_end] - mean[0]
            loss = loss.add(cloud_mask_loss(x_delta[0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t_end]))
        
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
        starting_time = time.time()
        all_data, path = batch
        path = path[0][0]

        _, cube_name = os.path.split(path)

        T = all_data.size()[4]
        t0 = 10 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0])

        start_model_evaluating = time.time()
        x_preds, x_deltas, means = self(all_data[:, :, :, :, :t0], prediction_count=T-t0, non_pred_feat=all_data[:,4:,:,:,t0+1:])
        print("model time: {0}".format(time.time() - start_model_evaluating))
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
            model_dir = os.getcwd() + "/model_instances/model_" + self.timestamp + "/"
            pred_dir = model_dir + "test_cubes/"
            
            model_pred_dir = pred_dir + "model_pred/"
            average_pred_dir = pred_dir + "average_pred/"
            last_pred_dir = pred_dir + "last_pred/"

            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)
                os.mkdir(model_pred_dir)
                os.mkdir(average_pred_dir)
                os.mkdir(last_pred_dir)

            num_context = round(all_data.shape[-1]/3)
            # Save avg predictions
            avg_start_time = time.time()
            avg_cube = mean_prediction(all_data[:, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
            print("avg time: {0}".format(time.time() - avg_start_time))

            np.savez(average_pred_dir+cube_name, avg_cube)
            # Save last cloud-free image predictions
            last_start_time = time.time()
            last_cube = last_prediction(all_data[:, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
            print("last time: {0}".format(time.time() - last_start_time))

            # Save our model prediction
            save_time = time.time()
            np.savez(last_pred_dir+cube_name, last_cube)
            print("save time: {0}".format(time.time() - save_time))

            np.savez(model_pred_dir+cube_name, x_preds)
            

            predictions = [average_pred_dir+cube_name, last_pred_dir+cube_name, model_pred_dir+cube_name]
            # Calculate ENS scores
            '''target_files = []
            with open(os.getcwd() + self.cfg["data"]["test_dir"] + '/target_files.txt', 'r') as filehandle:
                for line in filehandle:
                    # remove linebreak which is the last character of the string
                    cur = line[:-1]
                    target_files.append(cur)
            target_file = target_files[batch_idx]'''
            time_ens_score = time.time()
            scores = get_ENS(path, predictions)
            print("ENS score time: {0}".format(time.time() - time_ens_score))

            best_score = max(scores)
            with open(model_dir + "scores.csv", 'a') as filehandle:
                filehandle.write(str(scores[0]) + "," +str(scores[1]) + "," + str(scores[2]) + "," + str(best_score) + '\n')
            print("total time: {0}".format(time.time()-starting_time))