from math import floor
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
from Data.data_preparation import Earthnet_Dataset, Earthnet_NDVI_Dataset, Earthnet_Context_Dataset, prepare_train_data, prepare_test_data

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

# Create dirs if needed
if not os.path.exists(os.path.join(os.getcwd(), "Data", "all_data")):
    os.makedirs(os.path.join(os.getcwd(), "Data", "all_data"))
if not os.path.exists(os.path.join(os.getcwd(), "Data", "NDVI_data")):
    os.makedirs(os.path.join(os.getcwd(), "Data", "NDVI_data"))

# To build back the datasets
with open(os.path.join(os.getcwd(), "Data", "all_data", "train_data_paths.pkl"), "wb") as fp:
    pickle.dump(training_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "all_data", "val_1_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_1_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "all_data", "val_2_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_2_data.paths, fp)
with open(os.path.join(os.getcwd(), "Data", "all_data", "test_context_data_paths.pkl"), "wb") as fp:
    pickle.dump(test_data.context_paths, fp)
with open(os.path.join(os.getcwd(), "Data", "all_data", "test_target_data_paths.pkl"), "wb") as fp:
    pickle.dump(test_data.target_paths, fp)

# Create NDVI dataset (only using heavilly vegetated cubes)
NDVI_dataset = Earthnet_NDVI_Dataset(training_data.paths, cfg["data"]["mesoscale_cut"], device=device, veg_threshold=cfg["data"]["veg_threshold"])
NDVI_dataset.filter_non_vegetated()

NDVI_data = NDVI_dataset.paths
dataset_size = len(NDVI_data)

train_set = random.sample(NDVI_data, floor(dataset_size * 0.75))
rest = [x for x in NDVI_data if x not in train_set]
val_set = random.sample(rest, floor(dataset_size * 0.01))
if (len(val_set) == 0): # To work on small PC datasets
    val_set.append(train_set[-1])
    train_set = train_set[0:-1]
test_set = [x for x in rest if x not in val_set]

# To build back the datasets
with open(os.path.join(os.getcwd(), "Data", "NDVI_data", "train_data_paths.pkl"), "wb") as fp:
    pickle.dump(train_set, fp)
with open(os.path.join(os.getcwd(), "Data", "NDVI_data", "val_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_set, fp)
with open(os.path.join(os.getcwd(), "Data", "NDVI_data", "test_data_paths.pkl"), "wb") as fp:
    pickle.dump(test_set, fp)

print(len(train_set))
print(len(val_set))
print(len(test_set))
