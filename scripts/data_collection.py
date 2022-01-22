import sys
import os
import warnings
from os.path import join
from os.path import split
import random
import pickle
import glob
import numpy as np
from numpy import genfromtxt

sys.path.append(os.getcwd())

import argparse

training_samples = 23904

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-s',  '--source_folder', type=str, help='folder where the training data is')
parser.add_argument('-tf', '--test_folder', type=str, help='folder where the test data is', default=None)
parser.add_argument('-d',  '--dest_folder', type=str,  help='folder where pickles shoud be saved')
parser.add_argument('-td', '--training_data', type=int,  help='number of training data points')
parser.add_argument('-v1', '--val_1_data', type=int,  help='number of validation 1 data points')
parser.add_argument('-v2', '--val_2_data', type=int,  help='number of validation 2 data points', default=-1)
parser.add_argument('-se', '--seed', type=int,  help='seed for train/validation split', default=1)
parser.add_argument('-dc', '--data_cleaning', type=float,  help='level of data cleaning (-1: none, 0: cubes giving NaN, (0,1]: min score)', default=-1)
args = parser.parse_args()

# Collect all cubes in the training set
train_files = glob.glob(join(os.getcwd(), args.source_folder) + '/**/*.npz', recursive=True)
train_files.sort()
if len(train_files) != training_samples:
    warnings.warn("Your training set is incomplete! You only have " + str(len(train_files)) + " samples instead of " + str(training_samples))

random.seed(args.seed)
random.shuffle(train_files)

training_data = train_files[:args.training_data]
val_1_data = train_files[args.training_data: args.training_data + args.val_1_data]
if args.val_2_data == -1:
    val_2_data = train_files[args.training_data + args.val_1_data:-1]
else:
    val_2_data = train_files[args.training_data + args.val_1_data:args.training_data + args.val_1_data+args.val_2_data]

# Clean data
if args.data_cleaning != -1:
    baseline_scores = genfromtxt(join(os.getcwd(), "Data", "scores_last_frame.csv"), delimiter=',')
    with open(join(os.getcwd(), "Data", "last_frame_data_paths.pkl"),'rb') as f:
        old_train_paths = pickle.load(f)
    old_train_paths = [path.split()[-1] for path in old_train_paths]

    # Collect samples yielding NaN or below threshold
    bad_cubes = []
    for i in range(len(baseline_scores)):
        if np.isnan(baseline_scores[i, -1]):
            bad_cubes.append(split(old_train_paths[i])[-1])
    if args.data_cleaning > 0:
        for i in range(len(baseline_scores)):
            if baseline_scores[i, -1] < args.data_cleaning:
                bad_cubes.append(split(old_train_paths[i])[-1])
    
    # Discard bad samples
    training_data = [f for f in training_data if split(f)[-1] not in bad_cubes]

# Create pickle dir if needed
if not os.path.exists(join(os.getcwd(), args.dest_folder)):
    os.makedirs(join(os.getcwd(), args.dest_folder))

# To build back the datasets
with open(join(os.getcwd(), args.dest_folder, "train_data_paths.pkl"), "wb") as fp:
    pickle.dump(training_data, fp)
with open(join(os.getcwd(), args.dest_folder, "val_1_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_1_data, fp)
with open(join(os.getcwd(), args.dest_folder, "val_2_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_2_data, fp)

if args.test_folder is not None:
    # This folder name should indicate what test set we want (iid/ood/extreme/seasoanl)
    test_set = args.test_folder.split('/')[-2]
    
    test_context_files = []
    test_target_files = []
    for path, subdirs, files in os.walk(join(os.getcwd(), args.test_folder)):
        for name in files:
            if '.npz' in name:
                full_name = join(path, name)
                if 'context' in full_name:
                    test_context_files.append(full_name)
                elif 'target' in full_name:
                    test_target_files.append(full_name)

    # Sort file names just in case (so we glue together the right context & target)
    test_context_files.sort()
    test_target_files.sort()
    with open(join(os.getcwd(), args.dest_folder, test_set+"_context_data_paths.pkl"), "wb") as fp:
        pickle.dump(test_context_files, fp)
    with open(join(os.getcwd(), args.dest_folder, test_set+"_target_data_paths.pkl"), "wb") as fp:
        pickle.dump(test_target_files, fp)

print("Done")