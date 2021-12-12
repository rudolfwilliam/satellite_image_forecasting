from logging import Logger
import sys
import os
import numpy as np
import random
from shutil import copy2
from os import listdir
import pickle
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
from drought_impact_forecasting.models.Conv_model import Conv_model
from drought_impact_forecasting.models.Peephole_LSTM_model import Peephole_LSTM_model
from Data.data_preparation import Earthnet_Dataset, Earthnet_Test_Dataset, prepare_test_data
from scripts.callbacks import WandbTest_callback

import wandb
from datetime import datetime

def main():
    args, cfg = command_line_parser(mode = 'validate')

    model_path = os.path.join(cfg['path_dir'], "files", "runtime_model")
    models = listdir(model_path)
    models.sort()
    model_path = os.path.join(model_path , models[args.me])
    # to check that it's the last model

    print("testing experiment {0}".format(args.rn))
    print("testing model at epoch {0}".format(args.me))

    if not cfg["training"]["offline"]:
        wandb.login()

    # GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='test', offline=True)
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    # Use this also for NDVI
    with open(os.path.join(cfg['path_dir'], "files", "val_2_data_paths.pkl"),'rb') as f:
        val_2_path_list = pickle.load(f)

    try:
        with open(os.path.join(os.getcwd(), "Data", cfg["data"]["pickle_dir"], "test_context_data_paths.pkl"),'rb') as f:
            test_context_data = pickle.load(f)
        with open(os.path.join(os.getcwd(), "Data", cfg["data"]["pickle_dir"], "test_target_data_paths.pkl"),'rb') as f:
            test_target_data = pickle.load(f)
        test_data = Earthnet_Test_Dataset(test_context_data, test_target_data, cfg["data"]["mesoscale_cut"], device=device)
    except:
        test_data = prepare_test_data(cfg["data"]["mesoscale_cut"],cfg["data"]["test_dir"],device)

    test_data_loader = DataLoader(test_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["test_batch_size"], 
                                  drop_last=False)
    
    callbacks = WandbTest_callback(args.rn)

    #setup Trainer
    trainer = Trainer(  max_epochs=cfg["training"]["epochs"], 
                        logger=wandb_logger,
                        log_every_n_steps = min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["train_batch_size"]),
                        devices = cfg["training"]["devices"], 
                        accelerator=cfg["training"]["accelerator"],
                        callbacks=[callbacks])

    #setup Model
    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    elif args.model_name == "Conv_model":
        model = Conv_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    elif args.model_name == "Peephole_LSTM_model":
        model = Peephole_LSTM_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise ValueError("The specified model name is invalid.")

    # Run validation
    trainer.test(model, test_data_loader)

    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()