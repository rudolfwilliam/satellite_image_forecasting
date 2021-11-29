from logging import Logger
import sys
import os
import numpy as np
import random
from shutil import copy2
import pickle
import time

from torch.utils import data

#from pytorch_lightning.accelerators import accelerato
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
from drought_impact_forecasting.models.Transformer_model import Transformer_model
from drought_impact_forecasting.models.Baseline_model import Last_model
from drought_impact_forecasting.models.Conv_model import Conv_model
from Data.data_preparation import Earthnet_Context_Dataset, prepare_train_data, prepare_test_data
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

    wandb.init(entity="eth-ds-lab", project="drought_impact_forecasting-scripts")
    # Store the model to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))
    copy2(os.getcwd() + "/config/" + args.model_name + ".json", os.path.join(wandb.run.dir, args.model_name + ".json"))
    # GPU handling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("GPU count: {0}".format(gpu_count))

    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group=args.model_name, job_type='train', offline=True)
    
    
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    training_data, val_1_data, val_2_data = prepare_train_data(cfg["data"]["mesoscale_cut"],
                                                         cfg["data"]["train_dir"],
                                                         device = device,
                                                         training_samples=cfg["training"]["training_samples"],
                                                         val_1_samples=cfg["training"]["val_1_samples"],
                                                         val_2_samples=cfg["training"]["val_2_samples"])
    
    train_dataloader = DataLoader(training_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["train_batch_size"],
                                  shuffle=True, 
                                  drop_last=False)

    val_1_dataloader = DataLoader(val_1_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["val_1_batch_size"], 
                                  drop_last=False)

    val_2_dataloader = DataLoader(val_2_data, 
                                  num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["val_2_batch_size"], 
                                  drop_last=False)
    # To build back the datasets
    with open(os.path.join(wandb.run.dir, "train_data_paths.pkl"), "wb") as fp:
        pickle.dump(training_data.paths, fp)
    with open(os.path.join(wandb.run.dir, "val_1_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_1_data.paths, fp)
    with open(os.path.join(wandb.run.dir, "val_2_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_2_data.paths, fp)
    # Load model Callbacks
    wd_callbacks = WandbTrain_callback()
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
    elif args.model_name == "Transformer_model":
        model = Transformer_model(cfg)
    elif args.model_name == "Conv_model":
        model = Conv_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    # Run training
    trainer.fit(model, train_dataloader, val_1_dataloader)
    
    # Train on context frames of val2/test data
    if cfg["training"]["use_context"]:

        test_data = prepare_test_data(cfg["data"]["mesoscale_cut"],cfg["data"]["test_dir"],device)
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
