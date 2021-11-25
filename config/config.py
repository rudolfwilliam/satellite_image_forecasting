import argparse
import os
import json

def check_model_exists(name):
    if name not in ["LSTM_model", "Transformer_model", 'Conv_model']:
        raise ValueError("The specified model name is invalid.")

def command_line_parser(mode = "train"):
    """"
    Returns:
        args       -- command line arguments
        cfg (dict) -- configurations from a config file depending on the --model_name argument
    """
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    if mode == 'train':
        parser.add_argument('--model_name', type=str, default='LSTM_model', choices=['LSTM_model', 'Transformer_model', 'Conv_model'], help='frame prediction architecture')
        args = parser.parse_args()
        check_model_exists(args.model_name)
        cfg = json.load(open(os.getcwd() + "/config/" + args.model_name + ".json", 'r'))

    
    if mode == 'validate':
        parser.add_argument('--model_name', type=str, default='LSTM_model', choices=['LSTM_model', 'Transformer_model', 'Conv_model'], help='frame prediction architecture')
        parser.add_argument('--ts', type=str, help='timestamp of the model to validate: deprecated')
        parser.add_argument('--rn', type=str, help='wandb run name to validate')
        args = parser.parse_args()
        check_model_exists(args.model_name)
        try:
            dir_path = find_dir_path(args.rn)
            cfg = json.load(open(os.path.join(dir_path, "files",  args.model_name + ".json"), 'r'))
            cfg['path_dir'] = dir_path
        except:
            raise ValueError("The timestamp doesn't exist.")

    return args, cfg

def find_dir_path(wandb_name):
    dir_path = os.path.join(os.getcwd(), "wandb")

    runs = []
    for path, subdirs, files in os.walk(dir_path):
        for dir_ in subdirs:
            # Ignore any licence, progress, etc. files
            if os.path.isfile(os.path.join(dir_path,dir_, "files", "run_name.txt")):
                with open(os.path.join(dir_path,dir_, "files",  "run_name.txt"),'r') as f:
                    if (f.read() == wandb_name):
                        return os.path.join(dir_path,dir_)
    raise ValueError("The name doesn't exist.")


def read_config(path):
    cfg = json.load(open(path, 'r'))
    return cfg