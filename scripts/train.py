import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from config.config import command_line_parser
from drought_impact_forecasting.models.base_model import Base_model

# Here I would start simply with 1 datacube (see Data folder) just rgb values
# to see if the network is learning properly
def load_train_data():
    return [42, 1]

def main():
    args, cfg = command_line_parser()
    # Start just with a GAN, once it works we can make a VAE-GAN
    training_data = load_train_data()
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=cfg["training"]["batch_size"],
                                               shuffle=True, drop_last=True)

    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    trainer = Trainer(train_dataloader)

    if args.model_name == "base_model":
        model = Base_model(cfg)
    else:
        raise ValueError("The specified model name is invalid.")

    trainer.fit(model)
    #trainer.test(model)

if __name__ == "__main__":
    main()