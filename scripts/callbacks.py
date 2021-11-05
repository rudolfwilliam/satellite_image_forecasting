import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os
from os import path
import numpy as np
from pathlib import Path
from Data.data_preparation import prepare_data
import json



class Prediction_Callback(pl.Callback):
    def __init__(self, ms_cut, train_dir, test_dir, dataset, print_predictions):
        self.sample = dataset.__getitem__(0)
        self.print_predictions = print_predictions
        self.epoch = 0
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.top_dir = self.path + "/Predictions/"
        self.pred_dir = "predictions/"
        self.gt_dir = "ground_truth/"
        self.delta_dir = "deltas/"
        self.imgs_dir = "imgs/"

        # Set up Prediction directory structure if necessary
        if not path.isdir(self.top_dir):
            os.mkdir(self.top_dir)
        if not path.isdir(path.join(self.top_dir, self.pred_dir)):
            os.mkdir(path.join(self.top_dir, self.pred_dir))
        if not path.isdir(path.join(self.top_dir, self.gt_dir)):
            os.mkdir(path.join(self.top_dir, self.gt_dir))
        if not path.isdir(path.join(self.top_dir, self.pred_dir, self.delta_dir)):
            os.mkdir(path.join(self.top_dir, self.pred_dir, self.delta_dir))
        if not path.isdir(path.join(self.top_dir, self.pred_dir, self.imgs_dir)):
            os.mkdir(path.join(self.top_dir, self.pred_dir, self.imgs_dir))

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: "Optional" = None
    ) -> None:
        if self.print_predictions:
            if not os.path.exists(self.top_dir):
                self.__create_dir_structure()
            # take 10 context and predict 1
            pred, delta_pred, mean = trainer.model(torch.from_numpy(np.expand_dims(self.sample[:, :, :, :10], axis=0)))
            metrics = trainer.callback_metrics
            metrics['train_loss'] = [float(metrics['train_loss'])]
            metrics['lr'] = [float(metrics['lr'])]

            pre_pred = np.flip(pred[0, :3, :, :].detach().numpy().transpose(1, 2, 0).astype(float), -1)
            delta = np.flip(delta_pred[0, :4, :, :].detach().numpy().transpose(1, 2, 0).astype(float), -1)

            # values need to be between 0 and 1
            cor_pred = np.clip(pre_pred, 0, 1)
            if self.epoch == 0:
                with open(self.top_dir + self.pred_dir + "metrics.json", 'w') as fp:
                    json.dump(metrics, fp)
            else:
                with open(self.top_dir + self.pred_dir + "metrics.json", "r+") as fp:
                    data = json.load(fp)
                    data['train_loss'] = data['train_loss'] + metrics['train_loss'] 
                    data['lr'] = data['lr'] + metrics['lr'] 
                    fp.seek(0)
                    json.dump(data, fp)            



            plt.imsave(self.top_dir + self.pred_dir + self.imgs_dir + str(self.epoch) + "_pred.png", cor_pred)
            # store different rgb values of delta separately
            for c, i in enumerate(["r", "g", "b", "i"]):
                plt.imshow(delta[:, :, c])
                plt.colorbar()
                plt.savefig(self.top_dir + self.pred_dir + self.delta_dir + str(self.epoch) + "_delta_pred_" + i + ".png")
                plt.close()
            # in the very first epoch, store ground truth
            if self.epoch == 0:
                plt.imsave(self.top_dir + self.gt_dir + str(self.epoch) + "_gt.png", np.clip(np.flip(self.sample[:3, :, :, 10].detach().numpy().
                                                                transpose(1, 2, 0).astype(float), -1),0,1))
                
                # ground truth delta
                delta_gt = self.sample[:4, :, :, 10] - mean
                for c, i in enumerate(["r", "g", "b", "i"]):
                    plt.imshow(np.flip(delta_gt[0].detach().numpy().transpose(1, 2, 0).astype(float), -1)[:, :, c])
                    plt.colorbar()
                    plt.savefig(self.top_dir + self.gt_dir + str(self.epoch) + "_delta_gt_" + i + ".png")
                    plt.close()

            self.epoch += 1

    def __create_dir_structure(self):
        for sub_dir in [self.pred_dir, self.gt_dir]:
            if sub_dir == self.pred_dir:
                for sub_sub_dir in [self.delta_dir, self.imgs_dir]:
                    os.makedirs(self.top_dir + self.pred_dir + sub_sub_dir, exist_ok=True)
            else:
                os.makedirs(self.top_dir + self.gt_dir, exist_ok=True)



