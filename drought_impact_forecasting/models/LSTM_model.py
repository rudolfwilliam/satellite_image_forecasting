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
        Base prediction model. It is roughly based on the convolutional LSTM architecture.
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
                               dilation_rate=self.cfg["model"]["dilation_rate"],
                               hidden_dim=[hidden_channels] * n_layers,
                               kernel_size=(self.cfg["model"]["kernel"][0], self.cfg["model"]["kernel"][1]),
                               num_layers=n_layers,
                               num_conv_layers=self.cfg["model"]["num_conv_layers"],
                               num_conv_layers_mem=self.cfg["model"]["num_conv_layers_mem"],
                               batch_first=False)

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
        

        _, x_delta, mean = self(all_data[:, :, :, :, :t0])
        delta = all_data[:, :4, :, :, t0] - mean[0]
        loss = cloud_mask_loss(x_delta[0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t0])

        for t_end in range(t0 + 1, T): # this iterates with t_end = t0, ..., T-1
            _, x_delta, mean = self(all_data[:, :, :, :, :t_end])
            delta = all_data[:, :4, :, :, t_end] - mean[0]
            loss = loss.add(cloud_mask_loss(x_delta[0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t_end]))
        
        logs = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    # We could try early stopping here later on
    def validation_step(self, batch, batch_idx):
        '''
            The validation step also uses the L2 loss, but on a prediction of all non-context images
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
        t0 = 10 # no. of pics we start with

        _, x_delta, mean = self(all_data[:, :, :, :, :t0], prediction_count=T-t0, non_pred_feat=all_data[:,4:,:,:,t0+1:])
        
        delta = all_data[:, :4, :, :, t0] - mean[0]
        loss = cloud_mask_loss(x_delta[0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t0])
        
        for t_end in range(t0 + 1, T): # this iterates with t_end = t0, ..., T-1
            delta = all_data[:, :4, :, :, t_end] - mean[t_end-t0]
            loss = loss.add(cloud_mask_loss(x_delta[t_end-t0], delta, all_data[:,cloud_mask_channel:cloud_mask_channel+1, :,:,t_end]))

        logs = {'val_loss': loss}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def test_step(self, batch, batch_idx):
        '''
            The test step takes the test data and makes predictions.
            They are then evaluated using the ENS score.
        '''
        #starting_time = time.time()
        all_data, path = batch

        T = all_data.size()[4]
        t0 = 10 # no. of pics we start with
        l2_crit = nn.MSELoss()
        loss = torch.tensor([0.0])

        num_context = round(all_data.shape[-1]/3)

        # Create necessary directories
        model_dir = os.getcwd() + "/model_instances/model_" + self.timestamp + "/"
        pred_dir = model_dir + "test_cubes/"
        if not os.path.isdir(pred_dir):
            os.mkdir(pred_dir)

        if path[0] == "no target": # We use the validation test set
            model_val_pred_dir = pred_dir + "model_val_pred/"
            target_dir = pred_dir + "val_targets/"
            average_pred_dir = pred_dir + "val_average_pred/"
            last_pred_dir = pred_dir + "val_last_pred/"

            if not os.path.isdir(model_val_pred_dir):
                os.mkdir(model_val_pred_dir)
                os.mkdir(target_dir)
                os.mkdir(average_pred_dir)
                os.mkdir(last_pred_dir)
                with open(model_dir + "scores.csv", 'w') as filehandle:
                    filehandle.write("average, last, model, best\n")
                

            for i in range(len(path)):
                # Cut out 'target' data
                target = all_data[i,:5, :, :, t0:]
                target = np.array(target.cpu()).transpose(1,2,0,3)
                np.savez(target_dir+str(i), highresdynamic=target)

                # Avg prediction
                avg_cube = mean_prediction(all_data[i:i+1, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
                np.savez(average_pred_dir+str(i), highresdynamic=avg_cube)

                # Last cloud-free image
                last_cube = last_prediction(all_data[i:i+1, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
                np.savez(last_pred_dir+str(i), highresdynamic=last_cube)

                # Our model
                x_preds, x_deltas, means = self(all_data[i:i+1, :, :, :, :t0], prediction_count=T-t0, non_pred_feat=all_data[i:i+1,4:,:,:,t0+1:])
                x_preds = np.array(torch.cat(x_preds, axis=0).cpu()).transpose(2,3,1,0)
                np.savez(model_val_pred_dir+str(i), highresdynamic=x_preds)

                predictions = [average_pred_dir+str(i)+'.npz', last_pred_dir+str(i)+'.npz', model_val_pred_dir+str(i)+'.npz']
                # Calculate ENS scores
                scores = get_ENS(target_dir+str(i)+'.npz', predictions)

                best_score = max(scores)
                with open(model_dir + "scores.csv", 'a') as filehandle:
                    filehandle.write(str(scores[0]) + "," +str(scores[1]) + "," + str(scores[2]) + "," + str(best_score) + '\n')

        else: # We use the 'real' test set
            model_pred_dir = pred_dir + "model_pred/"
            average_pred_dir = pred_dir + "average_pred/"
            last_pred_dir = pred_dir + "last_pred/"
            if not os.path.isdir(model_pred_dir):
                os.mkdir(model_pred_dir)
                os.mkdir(average_pred_dir)
                os.mkdir(last_pred_dir)

            path = list(path)

            cube_name = [os.path.split(i)[1] for i in path]

            #start_model_evaluating = time.time()
            x_preds, x_deltas, means = self(all_data[:, :, :, :, :t0], prediction_count=T-t0, non_pred_feat=all_data[:,4:,:,:,t0+1:])
            #print("model time: {0}".format(time.time() - start_model_evaluating))
            # Add up losses across all timesteps
            for i in range(len(means)):
                delta = all_data[:,:4,:,:,t0+i] - means[i]
                loss = loss.add(l2_crit(x_deltas[i], delta))
            
            logs = {'test_loss': loss}
            self.log_dict(
                logs,
                on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            x_preds = np.array(torch.cat(x_preds, axis=0).cpu()).transpose(2,3,1,0)
            
            # Make all our predictions and save them
            # Store predictions ready for evaluation
            
            for i in range(len(path)):
                # Save avg predictions
                #avg_start_time = time.time()
                avg_cube = mean_prediction(all_data[i:i+1, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
                #print("avg time: {0}".format(time.time() - avg_start_time))

                np.savez(average_pred_dir+cube_name[i], highresdynamic=avg_cube)
                # Save last cloud-free image predictions
                #last_start_time = time.time()
                last_cube = last_prediction(all_data[i:i+1, 0:5, :, :, :num_context], mask_channel = 4, timepoints = num_context*2)
                #print("last time: {0}".format(time.time() - last_start_time))

                #save_time = time.time()
                np.savez(last_pred_dir+cube_name[i], highresdynamic=last_cube)
                #print("save time: {0}".format(time.time() - save_time))
                # Save our model prediction
                np.savez(model_pred_dir+cube_name[i], highresdynamic=x_preds[:,:,:,i*2*num_context:i*2*num_context+20])
            

                predictions = [average_pred_dir+cube_name[i], last_pred_dir+cube_name[i], model_pred_dir+cube_name[i]]
                # Calculate ENS scores
                #time_ens_score = time.time()
                scores = get_ENS(path[i], predictions)
                #print("ENS score time: {0}".format(time.time() - time_ens_score))

                best_score = max(scores)
                with open(model_dir + "scores.csv", 'a') as filehandle:
                    filehandle.write(str(scores[0]) + "," +str(scores[1]) + "," + str(scores[2]) + "," + str(best_score) + '\n')
                #print("total time: {0}".format(time.time()-starting_time))
