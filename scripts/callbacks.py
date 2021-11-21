import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os
from os import path
import numpy as np
from pathlib import Path
from Data.data_preparation import prepare_data
import json
import wandb



class Prediction_Callback(pl.Callback):
    def __init__(self, ms_cut, train_dir, test_dir, dataset, print_predictions, timestamp):
        self.sample, _ = dataset.__getitem__(0)
        self.print_predictions = print_predictions
        self.epoch = 0
        self.instance_folder = os.getcwd() + "/model_instances/model_" + timestamp
        self.runtime_model_folder = self.instance_folder + "/runtime_model"
        self.runtime_prediction = self.instance_folder + "/runtime_pred"
        self.r_pred = self.runtime_prediction + "/r"
        self.g_pred = self.runtime_prediction + "/g"
        self.b_pred = self.runtime_prediction + "/b"
        self.i_pred = self.runtime_prediction + "/i"
        self.img_pred = self.runtime_prediction + "/img"


        # Set up Prediction directory structure if necessary
        for dir_path in [self.instance_folder,
                         self.runtime_model_folder,self.runtime_prediction,
                         self.r_pred,self.g_pred,self.b_pred,self.i_pred,self.img_pred]:
            if not path.isdir(dir_path):
                os.mkdir(dir_path)
        
        if not os.path.isfile(self.instance_folder + "/scores.csv"):
            with open(self.instance_folder + "/scores.csv", 'w') as filehandle:
                filehandle.write("mad, ssim, ols, emd, score\n")
        
        self.channel_list = [self.r_pred, self.g_pred, self.b_pred, self.i_pred]
    
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: "Optional" = None
    ) -> None:

        torch.save(trainer.model.state_dict(), self.runtime_model_folder + "/model_"+str(self.epoch)+".torch")

        if self.print_predictions:
            # take 10 context and predict 1 (index from )
            preds, delta_preds, means = trainer.model(torch.unsqueeze(self.sample[:, :, :, :10], dim=0))
            metrics = trainer.callback_metrics
            metrics['train_loss'] = [float(metrics['train_loss'])]
            metrics['lr'] = [float(metrics['lr'])]
            metrics['online_val_loss'] = [float(metrics['online_val_loss'])]

            pre_pred = np.flip(preds[0][0, :3, :, :].detach().cpu().numpy().transpose(1, 2, 0).astype(float), -1)

            delta = np.flip(delta_preds[0][0, :4, :, :].detach().cpu().numpy().transpose(1, 2, 0).astype(float), -1)

            # values need to be between 0 and 1
            cor_pred = np.clip(pre_pred, 0, 1)
            if self.epoch == 0:
                with open(self.instance_folder  + "/metrics.json", 'w') as fp:
                    json.dump(metrics, fp)
            else:
                with open(self.instance_folder  + "/metrics.json", "r+") as fp:
                    data = json.load(fp)
                    data['train_loss'] = data['train_loss'] + metrics['train_loss'] 
                    data['lr'] = data['lr'] + metrics['lr']  
                    data['online_val_loss'] = data['online_val_loss'] + metrics['online_val_loss'] 
                    fp.seek(0)
                    json.dump(data, fp)            

            plt.imsave(self.img_pred + "/epoch_" + str(self.epoch) + ".png", cor_pred)
            ims = []
            # store different rgb values of delta separately
            if self.epoch == 0:
                plt.imsave(self.img_pred + "/gt.png", np.clip(np.flip(self.sample[:3, :, :, 9].detach().cpu().numpy().
                                                                transpose(1, 2, 0).astype(float), -1),0,1))
                
                # ground truth delta
                delta_gt = (self.sample[:4, :, :, 9] - means[0])[0]
                for i, c in enumerate(self.channel_list):
                    plt.imshow(np.flip(delta_gt.detach().cpu().numpy().transpose(1, 2, 0).astype(float), -1)[:, :, i])
                    plt.colorbar()
                    plt.savefig(c + "/gt.png")
                    plt.close()
                    ims.append(wandb.Image(plt.imread(c + "/gt.png"), 
                                           caption = "ground truth c: {1}".format(self.epoch, c[-1])))
                wandb.log({"Runtime Predictions":ims})

            ims = []
            for i, c in enumerate(self.channel_list):
                plt.imshow(delta[:, :, i])
                plt.colorbar()
                plt.savefig(c + "/epoch_" + str(self.epoch) + ".png")
                plt.close()
                ims.append(wandb.Image(plt.imread(c + "/epoch_" + str(self.epoch) + ".png"), 
                                       caption = "epoch: {0} c: {1}".format(self.epoch, c[-1])))

            wandb.log({"Runtime Predictions":ims})
            # in the very first epoch, store ground truth


            self.epoch += 1
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        return super().on_train_end(trainer, pl_module)





