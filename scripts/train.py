from logging import Logger
import sys
import os
from os import listdir
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
from drought_impact_forecasting.models.Transformer_model import Transformer_model
from drought_impact_forecasting.models.Baseline_model import Last_model
from drought_impact_forecasting.models.Conv_model import Conv_model
from Data.data_preparation import Earthnet_Dataset, Earthnet_Context_Dataset, prepare_train_data, prepare_test_data, Earth_net_DataModule
from callbacks import Prediction_Callback
from callbacks import WandbTrain_callback

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

    wandb.init(entity="eth-ds-lab", project="drought_impact_forecasting-scripts", config=cfg)
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

    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group=args.model_name, job_type='train', offline=True)
    
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    ENdataset = Earth_net_DataModule(data_dir = cfg["data"]["pickle_dir"], 
                                     train_batch_size = cfg["training"]["train_batch_size"],
                                     val_batch_size = cfg["training"]["val_1_batch_size"], 
                                     test_batch_size = cfg["training"]["val_2_batch_size"], 
                                     mesoscale_cut = cfg["data"]["mesoscale_cut"])
    
    # To build back the datasets for safety
    ENdataset.serialize_datasets(wandb.run.dir)
    

    # Load model Callbacks
    
    wd_callbacks = WandbTrain_callback(val_1_data = Earthnet_Dataset(ENdataset.val_1_path_list, cfg["data"]["mesoscale_cut"], device=device), cfg = cfg)
    # setup Trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                      logger=wandb_logger,
                      log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["train_batch_size"]),
                      devices = cfg["training"]["devices"],
                      accelerator=cfg["training"]["accelerator"],
                      callbacks=[wd_callbacks])

    # setup Model
    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
        #trainer.tune(model, train_dataloader)
    elif args.model_name == "Transformer_model":
        model = Transformer_model(cfg)
    elif args.model_name == "Peephole_LSTM_model":
        model = Peephole_LSTM_model(cfg)
    elif args.model_name == "Conv_model":
        model = Conv_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")
    
    # For training restarts
    # TODO: implement checkpoint callbacks and load from them
    if "path_dir" in cfg:
        model_path = os.path.join(cfg['path_dir'], "files", "runtime_model")
        models = listdir(model_path)
        models.sort()
        model_path = os.path.join(model_path , models[-1])
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Run training
    trainer.fit(model, ENdataset)
    
    # Train on context frames of val2/test data
    if cfg["training"]["use_context"]:

        test_data = prepare_test_data(cfg["data"]["mesoscale_cut"], cfg["data"]["test_dir"], device)
        context_data = Earthnet_Context_Dataset(test_data.context_paths, cfg["data"]["mesoscale_cut"], device)
        context_data = Earthnet_Context_Dataset(val_2_data.paths, cfg["data"]["mesoscale_cut"], device)
        context_dataloader = DataLoader(context_data, 
                             num_workers=cfg["training"]["num_workers"],
                             batch_size=cfg["training"]["train_batch_size"], 
                             drop_last=False)
        
        # This is ugly, but I couldn't find a better solution yet
        for i in range(cfg["training"]["epochs"]):
            trainer.fit(model, context_dataloader)
            torch.save(trainer.model.state_dict(), os.path.join(os.path.join(wandb.run.dir,"runtime_model"), "model_"+str(trainer.max_epochs+i)+".torch"))

    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()
