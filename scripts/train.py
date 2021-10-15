import sys
import os
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from config.config import command_line_parser
from drought_impact_forecasting.models.base_model import Base_model
from Data.data_preparation import prepare_data

# Here I would start simply with 1 datacube (see Data folder) just rgb values
# to see if the network is learning properly

def main():
    args, cfg = command_line_parser()
    # Start just with a GAN, once it works we can make a VAE-GAN
    training_data = prepare_data(True)
    test_data = prepare_data(False)
    #train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    trainer = Trainer()

    if args.model_name == "base_model":
        model = Base_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()