from logging import Logger
import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join
#from pytorch_lightning.accelerators import accelerator
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    print(timestamp)
    model_path = os.getcwd() + "/model_instances/model_"+timestamp+"/runtime_model"
    models = listdir(model_path)
    model_path = model_path + "/" + models[-1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_data, validation_data = prepare_data(cfg["data"]["mesoscale_cut"],
                                                  cfg["data"]["train_dir"],
                                                  device = device,
                                                  training_samples=cfg["training"]["training_samples"],
                                                  val_1_samples=cfg["training"]["val_1_samples"],
                                                  val_2_samples=cfg["training"]["val_2_samples"])
    test_data = prepare_data(cfg["data"]["mesoscale_cut"], 
                             cfg["data"]["test_dir"],
                             device = device)
    
    test_dataloader = DataLoader(test_data, 
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

    # setup trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"],
                        log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["train_batch_size"]),
                        devices = cfg["training"]["devices"], 
                        accelerator=cfg["training"]["accelerator"],
                        callbacks=[callbacks])

    model = LSTM_model(cfg, timestamp)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()
