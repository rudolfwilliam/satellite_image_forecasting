import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from pathlib import Path
from Data.data_preparation import prepare_data


class Prediction_Callback(pl.Callback):
    def __init__(self, ms_cut, train_dir, test_dir, print_predictions):
        self.sample = prepare_data(1, ms_cut, train_dir, test_dir)[0][0][0]
        self.print_predictions = print_predictions
        self.epoch = 0
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.top_dir = self.path + "/Predictions/"
        self.pred_dir = "predictions/"
        self.gt_dir = "ground_truth/"
        self.delta_dir = "deltas/"
        self.imgs_dir = "imgs/"

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: "Optional" = None
    ) -> None:
        if self.print_predictions:
            if not os.path.exists(self.top_dir):
                self.__create_dir_structure()
            # take 10 context and predict 1
            pred, delta_pred, mean = trainer.model(torch.from_numpy(np.expand_dims(self.sample[:, :, :, :10], axis=0)))

            pre_pred = np.flip(pred[0, :3, :, :].detach().numpy().transpose(1, 2, 0).astype(float), -1)
            delta = np.flip(delta_pred[0, :3, :, :].detach().numpy().transpose(1, 2, 0).astype(float), -1)

            # values need to be between 0 and 1
            cor_pred = np.clip(pre_pred, 0, 1)

            plt.imsave(self.top_dir + self.pred_dir + self.imgs_dir + str(self.epoch) + "_pred.png", cor_pred)
            # store different rgb values of delta separately
            for c, i in enumerate(["r", "g", "b"]):
                plt.imsave(self.top_dir + self.pred_dir + self.delta_dir + str(self.epoch) + "_delta_pred_" + i + ".png", delta[:, :, c])

            # in the very first epoch, store ground truth
            if self.epoch == 0:
                plt.imsave(self.top_dir + self.gt_dir + str(self.epoch) + "_gt.png", np.flip(self.sample[:3, :, :, 10].detach().numpy().
                                                                transpose(1, 2, 0).astype(float), -1))
                # ground truth delta
                delta_gt = self.sample[:4, :, :, 10] - mean
                for c, i in enumerate(["r", "g", "b"]):
                    plt.imsave(self.top_dir + self.gt_dir + str(self.epoch) + "_delta_gt_" + i + ".png", np.flip(delta_gt[0].
                                                                                detach().numpy().transpose(1, 2, 0).astype(float), -1)[:, :, c])

            self.epoch += 1

    def __create_dir_structure(self):
        for sub_dir in [self.pred_dir, self.gt_dir]:
            if sub_dir == self.pred_dir:
                for sub_sub_dir in [self.delta_dir, self.imgs_dir]:
                    os.makedirs(self.top_dir + self.pred_dir + sub_sub_dir, exist_ok=True)
            else:
                os.makedirs(self.top_dir + self.gt_dir, exist_ok=True)



