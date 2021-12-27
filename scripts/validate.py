from logging import Logger
import sys
import os
import numpy as np
import random
from shutil import copy2
from os import listdir
# from pytorch_lightning.accelerators import accelerator
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config.config import validate_line_parser
from drought_impact_forecasting.models.Peephole_LSTM_model import Peephole_LSTM_model
from Data.data_preparation import Earth_net_DataModule
from scripts.callbacks import WandbTest_callback

import wandb

def main():
    
    configs = validate_line_parser()

    print("validating experiment {0}".format(configs['run_name']))
    print("validating model at epoch {0}".format(configs['epoch_to_validate']))

    wandb.login()

    #GPU handling
    # print("GPU count: {0}".format(gpu_count))

    wandb_logger = WandbLogger(project='DS_Lab', job_type='test', offline=True)

    # Always use same val_2 data from Data folder

    ENdataset = Earth_net_DataModule(data_dir =configs['dataset_dir'], 
                                     train_batch_size = configs['batch_size'],
                                     val_batch_size = configs['batch_size'], 
                                     test_batch_size = configs['batch_size'], 
                                     use_real_test_set = configs['use_real_test_set'],
                                     mesoscale_cut = [39,41])
    
    callbacks = WandbTest_callback(configs['run_name'])

    #setup Trainer
    trainer = Trainer(logger=wandb_logger, callbacks=[callbacks])

    
    model = Peephole_LSTM_model.load_from_checkpoint(configs['model_path'])
    model.eval()
    #setup Model

    # Run validation
    trainer.test(model = model, dataloaders = ENdataset)

    wandb.finish()

if __name__ == "__main__":
    main()
