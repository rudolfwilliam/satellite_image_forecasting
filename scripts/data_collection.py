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

import argparse

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-s', '--source_folder', type=str, help='folder where data is')
parser.add_argument('-d', '--dest_folder', type=str,  help='folder where pickles shoud be saved')
parser.add_argument('-td', '--training_data', type=int,  help='folder where pickles shoud be saved')
parser.add_argument('-v1', '--val_1_data', type=int,  help='folder where pickles shoud be saved')
parser.add_argument('-v2', '--val_2_data', type=int,  help='folder where pickles shoud be saved', default=-1)
args = parser.parse_args()

train_files = []
for path, subdirs, files in os.walk(os.getcwd() + args.source_folder):    
    for name in files:
        # Ignore any licence, progress, etc. files
        if '.npz' in name:
            train_files.append(os.join(path, name))

train_files = random.shuffle(train_files)
training_data = train_files[:args.training_data]
val_1_data = train_files[args.training_data: args.training_data + args.val_1_data]
if args.val_2_data == -1:
    val_2_data = train_files[args.training_data + args.val_1_data: -1]
else:
    val_2_data = train_files[args.training_data + args.val_1_data:args.training_data + args.val_1_data+args.val_2_data]
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

print("Done")