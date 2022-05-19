import sys
import os
import warnings
import random
import pickle
import glob
import argparse
import numpy as np
sys.path.append(os.getcwd())
from os.path import join
from os.path import split
from numpy import genfromtxt

training_samples = 23904

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-s',  '--source_folder', type=str, help='folder where the training data is')
parser.add_argument('-st', '--source_target_folder', type=str, help='target folder (if we want to train on the seasonal data)', default=None)
parser.add_argument('-tf', '--test_folder', type=str, help='folder where the test data is', default=None)
parser.add_argument('-d',  '--dest_folder', type=str, help='folder where pickles shoud be saved')
parser.add_argument('-td', '--training_data', type=int, help='number of training data points')
parser.add_argument('-v1', '--val_1_data', type=int, help='number of validation 1 data points')
parser.add_argument('-v2', '--val_2_data', type=int, help='number of validation 2 data points', default=-1)
parser.add_argument('-se', '--seed', type=int, help='seed for train/validation split', default=1)
parser.add_argument('-t',  '--tile', type=str, help='specific tile name', default=None)
parser.add_argument('-dc', '--data_cleaning', type=float, help='level of data cleaning (-1: none, 0: cubes giving NaN, (0,1]: min score)', default=-1)
args = parser.parse_args()

# collect all cubes in the training set
train_files = glob.glob(join(os.getcwd(), args.source_folder) + '/**/*.npz', recursive=True)
train_files.sort()

# for 'small_data' dataset, remove user-dependent part of paths
if args.dest_folder == 'Data/small_data':
    for i in range(len(train_files)):
        train_files[i] = '/'.join('/'.join(train_files[i].split('\\')).split('/')[-4:])

train_files.sort()
if len(train_files) != training_samples:
    warnings.warn("Your training set is incomplete! You only have " + str(len(train_files)) + " samples instead of " + str(training_samples))

random.seed(args.seed)
random.shuffle(train_files)

training_data = train_files[:args.training_data]
val_1_data = train_files[args.training_data : args.training_data + args.val_1_data]
if args.val_2_data == -1:
    val_2_data = train_files[args.training_data + args.val_1_data:-1]
else:
    val_2_data = train_files[args.training_data + args.val_1_data:args.training_data + args.val_1_data+args.val_2_data]

if args.source_target_folder is not None:
    # collect all target cubes in the training set
    train_targ_files = glob.glob(join(os.getcwd(), args.source_target_folder) + '/**/*.npz', recursive=True)
    train_targ_files.sort()
    
    random.seed(args.seed)
    random.shuffle(train_targ_files)

    training_targ_data = train_targ_files[:args.training_data]
    val_1_targ_data = train_targ_files[args.training_data : args.training_data + args.val_1_data]
    if args.val_2_data == -1:
        val_2_targ_data = train_targ_files[args.training_data + args.val_1_data:-1]
    else:
        val_2_targ_data = train_targ_files[args.training_data + args.val_1_data:args.training_data + args.val_1_data+args.val_2_data]

# clean data
if args.data_cleaning != -1:
    baseline_scores = genfromtxt(join(os.getcwd(), "Data", "scores_last_frame.csv"), delimiter=',')
    with open(join(os.getcwd(), "Data", "last_frame_data_paths.pkl"),'rb') as f:
        old_train_paths = pickle.load(f)
    old_train_paths = [path.split()[-1] for path in old_train_paths]

    # collect samples yielding NaN or below threshold
    bad_cubes = []
    for i in range(len(baseline_scores)):
        if np.isnan(baseline_scores[i, -1]):
            bad_cubes.append(split(old_train_paths[i])[-1])
    if args.data_cleaning > 0:
        for i in range(len(baseline_scores)):
            if baseline_scores[i, -1] < args.data_cleaning:
                bad_cubes.append(split(old_train_paths[i])[-1])
    
    # discard bad samples
    training_data = [f for f in training_data if split(f)[-1] not in bad_cubes]

# create pickle dir if needed
if not os.path.exists(join(os.getcwd(), args.dest_folder)):
    os.makedirs(join(os.getcwd(), args.dest_folder))

# build back the datasets
with open(join(os.getcwd(), args.dest_folder, "train_data_paths.pkl"), "wb") as fp:
    pickle.dump(training_data, fp)
with open(join(os.getcwd(), args.dest_folder, "val_1_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_1_data, fp)
with open(join(os.getcwd(), args.dest_folder, "val_2_data_paths.pkl"), "wb") as fp:
    pickle.dump(val_2_data, fp)

if args.source_target_folder is not None:
    with open(join(os.getcwd(), args.dest_folder, "train_targ_data_paths.pkl"), "wb") as fp:
        pickle.dump(training_targ_data, fp)
    with open(join(os.getcwd(), args.dest_folder, "val_1_targ_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_1_targ_data, fp)
    with open(join(os.getcwd(), args.dest_folder, "val_2_targ_data_paths.pkl"), "wb") as fp:
        pickle.dump(val_2_targ_data, fp)

if args.test_folder is not None:
    # this folder name should indicate the desired test set (iid/ood/extreme/seasoanl)
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

    # filter a specific tile
    if args.tile is not None:
        test_context_files = [i for i in test_context_files if args.tile in i]
        test_target_files = [i for i in test_target_files if args.tile in i]

    # for 'small_data' dataset take out user-dependent part of paths
    if args.dest_folder == 'Data/small_data':
        for i in range(len(test_context_files)):
            test_context_files[i] = '/'.join('/'.join(test_context_files[i].split('\\')).split('/')[-6:])
            test_target_files[i] = '/'.join('/'.join(test_target_files[i].split('\\')).split('/')[-6:])

    # sort file names, just in case (glue together the right context & target)
    test_context_files.sort()
    test_target_files.sort()
    with open(join(os.getcwd(), args.dest_folder, test_set+"_context_data_paths.pkl"), "wb") as fp:
        pickle.dump(test_context_files, fp)
    with open(join(os.getcwd(), args.dest_folder, test_set+"_target_data_paths.pkl"), "wb") as fp:
        pickle.dump(test_target_files, fp)

print("Done")