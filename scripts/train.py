from logging import Logger
import sys
import os
import numpy as np
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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


def main():
    args, cfg = command_line_parser()
    if not cfg["training"]["offline"]:
        wandb.login()
    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, group='LSTM', job_type='train', offline=True)
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    training_data, test_data = prepare_data(cfg["training"]["training_samples"], cfg["data"]["mesoscale_cut"],
                        cfg["data"]["train_dir"], cfg["data"]["test_dir"])
    train_dataloader = DataLoader(training_data, num_workers=cfg["training"]["num_workers"],
                                  batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, num_workers=cfg["training"]["num_workers"], drop_last=False)

    # We might want to configure GPU, TPU, etc. usage here
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], logger=wandb_logger,
                      log_every_n_steps=min(cfg["training"]["log_steps"],
                                            cfg["training"]["training_samples"] / cfg["training"]["batch_size"]),
                      devices=cfg["training"]["devices"], accelerator=cfg["training"]["accelerator"],
                      callbacks=[Prediction_Callback(cfg["data"]["mesoscale_cut"], cfg["data"]["train_dir"],
                                                     cfg["data"]["test_dir"], cfg["training"]["print_predictions"])])

    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model, train_dataloader)

    torch.save(model.state_dict(), "Models/model.torch")

    trainer.test(model, test_dataloader)

    # We may have to add a floor/ceil function on the predictions
    # sometimes we get out of bound values!
    # trainer.predict(model, test_dataloader)

    if not cfg["training"]["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()
