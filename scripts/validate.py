from logging import Logger
import sys
import os
import numpy as np
import random
from shutil import copy2
from os import listdir
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
    args, cfg = command_line_parser(mode = 'validate')
    #filepath = os.getcwd() + cfg["project"]["model_path"]
    timestamp = args.ts
    print("Timestamp: {0}".format(timestamp))
    model_path = os.getcwd() + "/model_instances/model_"+timestamp+"/runtime_model"
    models = listdir(model_path)
    model_path = model_path + "/" + models[-1]
    # to check that it's the last model
    print("validating model {0}".format(models[-1]))

    if not cfg["training"]["offline"]:
        wandb.login()

    #GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("GPU count: {0}".format(gpu_count))

    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='train', offline=True)
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    # For now the seed is super important!! Otherwise we validate on the training set. TODO: more elegant solution
    training_data, val_1_data, val_2_data = prepare_data(cfg["data"]["mesoscale_cut"],
                                                  cfg["data"]["train_dir"],
                                                  device = device,
                                                  training_samples=cfg["training"]["training_samples"],
                                                  val_1_samples=cfg["training"]["val_1_samples"],
                                                  val_2_samples=cfg["training"]["val_2_samples"])
   
    val_2_dataloader = DataLoader(val_2_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["val_2_batch_size"], 
                                  drop_last=False)


    # Load model Callbacks
    callbacks = Prediction_Callback(cfg["data"]["mesoscale_cut"], 
                                    cfg["data"]["train_dir"],
                                    cfg["data"]["test_dir"], 
                                    training_data,
                                    cfg["training"]["print_predictions"],
                                    timestamp)

    #setup Trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                        logger=wandb_logger,
                        log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["train_batch_size"]),
                        devices = cfg["training"]["devices"], 
                        accelerator=cfg["training"]["accelerator"],
                        callbacks=[callbacks])

    #setup Model
    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg, timestamp)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("The specified model name is invalid.")

    # Run validation
    trainer.test(model, val_2_dataloader)

    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()
