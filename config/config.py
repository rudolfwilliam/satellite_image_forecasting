import argparse
import os
import json

def check_model_exists(name):
    if name not in ["LSTM_model", "Transformer_model"]:
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
        parser.add_argument('--model_name', type=str, default='LSTM_model', choices=['LSTM_model', 'Transformer_model'], help='frame prediction architecture')
        args = parser.parse_args()
        check_model_exists(args.model_name)
        cfg = json.load(open(os.getcwd() + "/config/" + args.model_name + ".json", 'r'))

    
    if mode == 'validate':
        parser.add_argument('--ts', type=str, help='timestamp of the model to validate')
        args = parser.parse_args()
        check_model_exists(args.model_name)
        try:
            cfg = json.load(open(os.getcwd() + "/model_instances/model_" + args.ts + "/Transformer_model.json", 'r'))
        except:
            raise ValueError("The timestamp doesn't exist.")

    return args, cfg

def read_config(path):
    cfg = json.load(open(path, 'r'))
    return cfg