from logging import Logger
import sys
import os
import numpy as np
from shutil import copy2
#from pytorch_lightning.accelerators import acceleratofrom pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config.config import command_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from Data.data_preparation import prepare_data
from scripts.callbacks import Prediction_Callback

import wandb
from datetime import datetime

def main():

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") # timestamp unique to this training instance
    print("Timestamp of the instance: " + timestamp)
    os.mkdir(os.getcwd() + "/model_instances/model_"+timestamp)
    copy2(os.getcwd() + "/config/LSTM_model.json", os.getcwd() + "/model_instances/model_"+timestamp + "/LSTM_model.json")

    args, cfg = command_line_parser(mode = 'train')
    print(args)

    if not cfg["training"]["offline"]:
        wandb.login()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print("GPU count: {0}".format(gpu_count))

    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='train', offline=True)
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    training_data, test_data = prepare_data(cfg["training"]["training_samples"], 
                                            cfg["data"]["mesoscale_cut"],
                                            cfg["data"]["train_dir"], 
                                            cfg["data"]["test_dir"],
                                            device = device)
    train_dataloader = DataLoader(training_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["batch_size"],
                                  shuffle=True, 
                                  drop_last=False)
    test_dataloader = DataLoader(test_data, 
                                num_workers = cfg["training"]["num_workers"],
                                batch_size = 1, 
                                drop_last = False)

    # We might want to configure GPU, TPU, etc. usage here
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                        logger=wandb_logger,
                        log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["batch_size"]),
                        devices = cfg["training"]["devices"], 
                        accelerator=cfg["training"]["accelerator"],
                        callbacks=[ Prediction_Callback(cfg["data"]["mesoscale_cut"], 
                                                        cfg["data"]["train_dir"],
                                                        cfg["data"]["test_dir"], 
                                                        training_data,
                                                        cfg["training"]["print_predictions"],
                                                        timestamp)],
                        gpus = gpu_count)

    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg, timestamp)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model, train_dataloader)

    #torch.save(model.state_dict(), "Models/model.torch")
    trainer.test(model, test_dataloader)

    # We may have to add a floor/ceil function on the predictions
    # sometimes we get out of bound values!
    # trainer.predict(model, test_dataloader)

    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()
