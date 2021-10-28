from logging import Logger
import sys
import os
from pytorch_lightning.accelerators import accelerator
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config.config import command_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from Data.data_preparation import prepare_data

import matplotlib.pylab as plt
import numpy as np

import wandb

def main():   
    # wandb.login()
    args, cfg = command_line_parser()
    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='train')
    pl.seed_everything(cfg["training"]["seed"])

    training_data, test_data = prepare_data(cfg["training"]["training_samples"], cfg["data"]["mesoscale_cut"])
    train_dataloader = DataLoader(training_data, num_workers = cfg["training"]["num_workers"], batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, num_workers = cfg["training"]["num_workers"], drop_last=False)

    model = LSTM_model(cfg)
    model.load_state_dict(torch.load("Models/model.torch"))
    
    pred_training_data = model(training_data.highres_dynamic[:,:,:,:,:])[0]
    #pred_training_data = model(training_data.highres_dynamic)[0]
    pred_training_data = model(test_data)

    idx = 1



    fig, axs = plt.subplots(1,2)
    #for idx 
    pic_data = np.flip(pred_training_data[idx,:3,:,:].detach().numpy().transpose(1,2,0),2)
    axs[0].imshow(pic_data)
    pic_pred = np.flip(training_data.highres_dynamic[idx,:3,:,:,29].detach().numpy().transpose(1,2,0), 2)
    axs[1].imshow(pic_pred)
    plt.show()


    channel = 2
    fig, axs = plt.subplots(1,2)
    #for idx 
    pic_pred = pred_training_data[idx,channel,:,:].detach().numpy()
    axs[0].imshow(pic_pred)
    pic_data = training_data.highres_dynamic[idx,channel,:,:,29].detach().numpy()
    axs[1].imshow(pic_data)
    plt.show()
    # We may have to add a floor/ceil function on the predictions
    # sometimes we get out of bound values!
    # trainer.predict(model, test_dataloader)

    # wandb.finish()

if __name__ == "__main__":
    main()