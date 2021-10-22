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

import wandb

def main():   
    wandb.login()
    args, cfg = command_line_parser()
    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='train')
    pl.seed_everything(cfg["training"]["seed"])

    training_data, test_data = prepare_data(cfg["training"]["training_samples"])
    train_dataloader = DataLoader(training_data, num_workers = cfg["training"]["num_workers"], batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, num_workers = cfg["training"]["num_workers"], drop_last=False)

    # We might want to configure GPU, TPU, etc. usage here
    trainer = Trainer(max_epochs = cfg["training"]["epochs"], logger = wandb_logger,
                        devices = cfg["training"]["devices"], accelerator = cfg["training"]["accelerator"])

    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

    # We may have to add a floor/ceil function on the predictions
    # sometimes we get out of bound values!
    # trainer.predict(model, test_dataloader)

    wandb.finish()

if __name__ == "__main__":
    main()