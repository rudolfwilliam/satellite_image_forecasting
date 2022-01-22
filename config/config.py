import argparse
from operator import index
import os
from os.path import join
import json

def train_line_parser():
    # Load default args from json
    cfg = json.load(open(os.getcwd() + "/config/Training.json", 'r'))

    # Parse settable args from terminal
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-tl', '--training_loss', type=str, default='l2', choices=['l1','l2','Huber'], help='loss function used for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('-bm', '--big_memory', type=str, default=None, help='big memory or small: t = ture, f = false')
    parser.add_argument('-nl', '--num_layers', type=int, default=None, help='number of layers')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=None, help='number of hidden channels')
    parser.add_argument('-ln', '--layer_normalization', type=str, default=None, help='layer normalization: t = true, f = false')
    parser.add_argument('-k',  '--kernel_size', type=int, default=None, help='convolution kernel size')
    parser.add_argument('-mk', '--mem_kernel_size', type=int, default=None, help='memory kernel size')
    parser.add_argument('-ft', '--future_training', type=int, default=None, help='future steps for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=None, help='starting learning rate')
    parser.add_argument('-lf', '--learning_factor', type=float, default=None, help='learning rate factor')
    parser.add_argument('-p',  '--patience', type=int, default=None, help='patience')
    parser.add_argument('-e',  '--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('-bf', '--baseline_function', type=str, default=None, choices=['mean_cube', 'last_frame', 'zeros'], help='baseline function')
    parser.add_argument('-pd', '--pickle_dir', type=str, default=None, help='directory with the desired pickle files')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to continue from')
    parser.add_argument('-fw', '--fake_weather', type=str, default=None, help='if true weather is masked: t = true, f = false')
    args = parser.parse_args()

    if args.batch_size is not None:
        cfg["training"]["train_batch_size"] = args.batch_size
        cfg["training"]["val_1_batch_size"] = args.batch_size
        cfg["training"]["val_2_batch_size"] = args.batch_size

    if args.big_memory is not None:
        if args.big_memory == "y" or args.big_memory == "Y" or args.big_memory == "T" or args.big_memory == "t":
            cfg["model"]["big_mem"] = True
        elif args.big_memory == "n" or args.big_memory == "N" or args.big_memory == "f" or args.big_memory == "F":
            cfg["model"]["big_mem"] = False
    
    if args.fake_weather is not None:
        if args.fake_weather == "y" or args.fake_weather == "Y" or args.fake_weather == "T" or args.fake_weather == "t":
            cfg["training"]["fake_weather"] = True
        elif args.fake_weather == "n" or args.fake_weather == "N" or args.fake_weather == "f" or args.fake_weather == "F":
            cfg["training"]["fake_weather"] = False
    
    if args.layer_normalization is not None:
        if args.layer_normalization == "y" or args.layer_normalization == "Y" or args.layer_normalization == "T" or args.layer_normalization == "t":
            cfg["model"]["layer_norm"] = True
        elif args.layer_normalization == "n" or args.layer_normalization == "N" or args.layer_normalization == "f" or args.layer_normalization == "F":
            cfg["model"]["layer_norm"] = False
    
    if args.hidden_channels is not None:
        cfg["model"]["hidden_channels"] = args.hidden_channels
    
    if args.kernel_size is not None:
        cfg["model"]["kernel"] = args.kernel_size

    if args.mem_kernel_size is not None:
        cfg["model"]["memory_kernel"] = args.mem_kernel_size

    if args.num_layers is not None:
        cfg["model"]["n_layers"] = args.num_layers

    if args.future_training is not None:
        cfg["model"]["future_training"] = args.future_training

    if args.learning_rate is not None:
        cfg["training"]["start_learn_rate"] = args.learning_rate

    if args.learning_factor is not None:
        cfg["training"]["lr_factor"] = args.learning_factor

    if args.patience is not None:
        cfg["training"]["patience"] = args.patience

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.baseline_function is not None:
        cfg["model"]["baseline"] = args.baseline_function

    if args.training_loss is not None:
        cfg["training"]["training_loss"] = args.training_loss
        
    if args.pickle_dir is not None:
        cfg["data"]["pickle_dir"] = args.pickle_dir

    cfg["training"]["checkpoint"] = args.checkpoint
    
    return cfg

def validate_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-cp','--check_point_path', type=str, default=None, help='checkpoint file to validate. Note: either use --checkpoint or --run_name, not both')
    parser.add_argument('-rn','--run_name', type=str, default=None, help='wandb run name to validate. Note: either use --checkpoint or --run_name, not both')
    parser.add_argument('-bs','--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-vd','--validation_dataset', type=str, default=None, help='path to the pickle folder of the validation set')
    parser.add_argument('-ts','--test_set', type=str, default='val_2', help='test split to use')
    parser.add_argument('-e', '--epoch_to_validate', type=int, default=-1, help='model epoch to test/validate')
    args = parser.parse_args()

    if args.run_name is not None and args.check_point_path is not None:
        raise ValueError("Either use --check_point_path or --run_name, not both!")

    if args.run_name is not None and args.check_point_path is None:
        dir_path = find_dir_path(args.run_name)
        model_path = join(dir_path, "files", "runtime_model")
        models = os.listdir(model_path)
        models.sort()
        args.epoch_to_validate = (args.epoch_to_validate + len(models)) % len(models)
        model_path = join(model_path , models[args.epoch_to_validate])
        if args.validation_dataset is None:
            cfg = json.load(open(join(dir_path, "files", "Training.json"), 'r'))
            dataset_dir = cfg["data"]["pickle_dir"]
        else:
            dataset_dir = args.validation_dataset
    
    if args.run_name is None and args.check_point_path is not None:
        model_path = args.check_point_path
        if args.validation_dataset is None:
            raise ValueError("When using a checkpoint outside of a folder, one must specify the dataset_directory")
        else:
            dataset_dir = args.validation_dataset

    configs = dict(
        model_path = model_path,
        dataset_dir = dataset_dir,
        run_name = args.run_name,
        epoch_to_validate = args.epoch_to_validate,
        batch_size = args.batch_size,
        test_set = args.test_set
    )
    return configs

def diagnosticate_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-cp','--check_point_path', type=str, default=None, help='checkpoint file to validate. Note: either use --checkpoint or --run_name, not both')
    parser.add_argument('-rn','--run_name', type=str, default=None, help='wandb run name to validate. Note: either use --checkpoint or --run_name, not both')
    parser.add_argument('-e', '--epoch_to_validate', type=int, default=-1, help='model epoch to test/validate')
    parser.add_argument('-td','--train_dataset', type=str, default=None, help='pickle file with train set')
    parser.add_argument('-tcd','--test_context_dataset', type=str, default=None, help='pickle file with context of a test set')
    parser.add_argument('-ttd','--test_target_dataset', type=str, default=None, help='pickle file with target of a test set')
    parser.add_argument('-id','--cube_index', type=int, default=0, help='index of cube to use')
    parser.add_argument('-a','--action', type=str, default=None, choices=['visualize', 'time_plot'], help='diagnosticate kind to run')
    args = parser.parse_args()

    if args.run_name is not None and args.check_point_path is not None:
        raise ValueError("Either use --check_point_path or --run_name, not both!")

    if args.run_name is not None and args.check_point_path is None:
        dir_path = find_dir_path(args.run_name)
        model_path = join(dir_path, "files", "runtime_model")
        models = os.listdir(model_path)
        models.sort()
        args.epoch_to_validate = (args.epoch_to_validate + len(models)) % len(models)
        model_path = join(model_path , models[args.epoch_to_validate])
    
    if args.run_name is None and args.check_point_path is not None:
        model_path = args.check_point_path
    
    if args.test_context_dataset is not None and args.test_target_dataset is None or \
        args.test_context_dataset is None and args.test_target_dataset is not None:
        raise ValueError("When test data cube you must supply a context and target")

    configs = dict(
        model_path = model_path,
        run_name = args.run_name,
        epoch_to_validate = args.epoch_to_validate,
        train_data = args.train_dataset,
        test_context_data = args.test_context_dataset,
        test_target_data = args.test_target_dataset,
        index = args.cube_index,
        action = args.action
    )
    return configs

def find_dir_path(wandb_name):
    dir_path = os.path.join(os.getcwd(), "wandb")

    runs = []
    for path, subdirs, files in os.walk(dir_path):
        for dir_ in subdirs:
            # Ignore any licence, progress, etc. files
            if os.path.isfile(join(dir_path,dir_, "files", "run_name.txt")):
                with open(join(dir_path,dir_, "files",  "run_name.txt"),'r') as f:
                    if (f.read() == wandb_name):
                        return join(dir_path,dir_)
    raise ValueError("The name doesn't exist.")

def read_config(path):
    cfg = json.load(open(path, 'r'))
    return cfg