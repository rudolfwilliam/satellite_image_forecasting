from logging import Logger
import sys
import os
import json
import numpy as np
import random
from shutil import copy2
import pickle
import time

from torch.utils import data

# from pytorch_lightning.accelerators import accelerator
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config.config import command_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from drought_impact_forecasting.models.Peephole_LSTM_model import Peephole_LSTM_model
from drought_impact_forecasting.models.NDVI_Peephole_LSTM_model import NDVI_Peephole_LSTM_model
from drought_impact_forecasting.models.Transformer_model import Transformer_model
from drought_impact_forecasting.models.Baseline_model import Last_model
from drought_impact_forecasting.models.Conv_model import Conv_model
from Data.data_preparation import Earthnet_Dataset, Earthnet_Context_Dataset, prepare_train_data, prepare_test_data, Earthnet_NDVI_Dataset
from callbacks import Prediction_Callback,WandbTrain_callback,SDVI_Train_callback
import wandb
from datetime import datetime

def main():

    #TODO: Make this modular so that it works both for LSTM and Transformer

    args, cfg = command_line_parser(mode='train')
    print(args)

    if not cfg["training"]["offline"]:
        os.environ["WANDB_MODE"]="online"
    else:
        os.environ["WANDB_MODE"]="offline"

    wandb.init(entity="eth-ds-lab", project="NDVI-prediction")
    # Store the model to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))

    with open(os.path.join(wandb.run.dir, args.model_name + ".json"), 'w') as fp:
        json.dump(cfg, fp)
    
    # GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandbLogger(project='NDVI-prediction', config=cfg, group=args.model_name, job_type='train', offline=True)
    
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    try:
        # Try to load data paths quickly from pickle file
        with open(os.path.join(os.getcwd(), "Data", cfg["data"]["pickle_dir"], "train_data_paths.pkl"),'rb') as f:
            training_data = pickle.load(f)
        training_data = Earthnet_NDVI_Dataset(training_data, cfg["data"]["mesoscale_cut"], device=device)
        with open(os.path.join(os.getcwd(), "Data", cfg["data"]["pickle_dir"], "val_1_data_paths.pkl"),'rb') as f:
            val_1_data = pickle.load(f)
        val_1_data = Earthnet_NDVI_Dataset(val_1_data, cfg["data"]["mesoscale_cut"], device=device)
        with open(os.path.join(os.getcwd(), "Data", cfg["data"]["pickle_dir"], "val_2_data_paths.pkl"),'rb') as f:
            val_2_data = pickle.load(f)
        val_2_data = Earthnet_NDVI_Dataset(val_2_data, cfg["data"]["mesoscale_cut"], device=device)
    except:
        training_data, val_1_data, val_2_data = prepare_train_data(cfg["data"]["mesoscale_cut"],
                                                         cfg["data"]["train_dir"],
                                                         device = device,
                                                         training_samples=cfg["training"]["training_samples"],
                                                         val_1_samples=cfg["training"]["val_1_samples"],
                                                         val_2_samples=cfg["training"]["val_2_samples"],
                                                         undersample=False)
    
    train_dataloader = DataLoader(training_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["train_batch_size"],
                                  shuffle=True, 
                                  drop_last=False)

    val_1_dataloader = DataLoader(val_1_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["val_1_batch_size"], 
                                  drop_last=False)

    '''val_2_dataloader = DataLoader(val_2_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["val_2_batch_size"], 
                                  drop_last=False)'''

    # To build back the datasets
    with open(os.path.join(wandb.run.dir, "train_data_paths.pkl"), "wb") as fp:
        pickle.dump(training_data.paths, fp)
    with open(os.path.join(wandb.run.dir, "val_1_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_1_data.paths, fp)
    with open(os.path.join(wandb.run.dir, "val_2_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_2_data.paths, fp)

    callbacks = SDVI_Train_callback()
    # setup Trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                      logger=wandb_logger,
                      log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["train_batch_size"]),
                      devices = cfg["training"]["devices"],
                      accelerator=cfg["training"]["accelerator"],
                      callbacks = callbacks)

    # setup Model

    if args.model_name == "NDVI_Peephole_LSTM_model":
        model = NDVI_Peephole_LSTM_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    # Run training
    trainer.fit(model, train_dataloader, val_1_dataloader)
    
    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()