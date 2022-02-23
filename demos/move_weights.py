from gc import callbacks
import sys
import os
import numpy as np
from matplotlib import gridspec
import sys
import os
from os import listdir
from pathlib import Path
import json
import wandb
import random
import pytorch_lightning as pl
sys.path.append(os.getcwd())
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config.config import train_line_parser
from drought_impact_forecasting.models.EN_model import EN_model
from Data.data_preparation import Earth_net_DataModule
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os
from os import path
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from drought_impact_forecasting.models.utils.utils import mean_prediction, last_prediction, ENS
import wandb

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from load_model_data import *

    model = load_model()
def main():
    filename = None
    truth, context, target, npf = load_data_point()
    model = load_model()
    callback = Fake_Callback()
    trainer = Trainer(callbacks=callback, 
                      num_sanity_val_steps=1)
    trainer.fit(model)

class Fake_Callback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.save_checkpoint("BEST_MODEL")
        return super().on_train_start(trainer, pl_module)
main()