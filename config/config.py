import argparse
import os
import json


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
        parser.add_argument('--model_name', type=str, default='LSTM_model', choices=['LSTM_model'], help='CNN architecture')
        args = parser.parse_args()
        if args.model_name == "LSTM_model":
            cfg = json.load(open(os.getcwd() + "/config/LSTM_model.json", 'r'))
        else:
            raise ValueError("The specified model name is invalid.")
    
    if mode == 'validate':
        parser.add_argument('--ts', type=str, help='timestamp of the model to validate')
        args = parser.parse_args()
        try:
            cfg = json.load(open(os.getcwd() + "/model_instances/model_"+args.ts+"/LSTM_model.json", 'r'))
        except:
            raise ValueError("The timestamp doesn't exist.")
    
    return args, cfg

def read_config(path):
    cfg = json.load(open(path, 'r'))
    return cfg