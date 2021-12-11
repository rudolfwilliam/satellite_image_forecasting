import sys
import os
import json
import random
import time
import pickle

import torch
from torch.utils import data
import pytorch_lightning as pl

sys.path.append(os.getcwd())

from config.config import command_line_parser
from Data.data_preparation import Earthnet_Dataset, Earthnet_Context_Dataset, prepare_train_data, prepare_test_data

args, cfg = command_line_parser(mode='train')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(cfg["training"]["seed"])
pl.seed_everything(cfg["training"]["seed"], workers=True)

training_data, val_1_data, val_2_data = prepare_train_data(cfg["data"]["mesoscale_cut"],
                                                        cfg["data"]["train_dir"],
                                                        device = device,
                                                        training_samples=cfg["training"]["training_samples"],
                                                        val_1_samples=cfg["training"]["val_1_samples"],
                                                        val_2_samples=cfg["training"]["val_2_samples"],
                                                        undersample=False)
test_data = prepare_test_data(cfg["data"]["mesoscale_cut"],cfg["data"]["test_dir"],device)

# To build back the datasets
with open(os.path.join(os.getcwd(), "Data", "train_data_paths.pkl"), "wb") as fp:
    pickle.dump(training_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "val_1_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_1_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "val_2_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_2_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "test_context_data_paths.pkl"), "wb") as fp:
    pickle.dump(test_data.context_paths, fp)
with open(os.path.join(os.getcwd(), "Data", "test_target_data_paths.pkl"), "wb") as fp:
    pickle.dump(test_data.target_paths, fp)