import sys
import os
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from config.config import command_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from Data.data_preparation import prepare_data

# Here I would start simply with 1 datacube (see Data folder) just rgb values
# to see if the network is learning properly

def main():   

    args, cfg = command_line_parser()
    pl.seed_everything(cfg["training"]["seed"])

    training_data, test_data = prepare_data(cfg["training"]["training_samples"])
    train_dataloader = DataLoader(training_data, num_workers = cfg["training"]["num_workers"], batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, num_workers = cfg["training"]["num_workers"], batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)

    trainer = Trainer(max_epochs = cfg["training"]["epochs"]) # We might want to configure GPU, TPU, etc. usage here

    if args.model_name == "LSTM_model":
        model = LSTM_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

    # We may have to add a floor/ceil function on the predictions
    # sometimes we get out of bound values!
    # trainer.predict(model, test_dataloader)

if __name__ == "__main__":
    main()