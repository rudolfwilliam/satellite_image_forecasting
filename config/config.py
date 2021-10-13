import argparse
import os
import json


def command_line_parser():
    """"
    Returns:
        args       -- command line arguments
        cfg (dict) -- configurations from a config file depending on the --model_name argument
    """
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_name', type=str, default='base_model', choices=['base_model'], help='CNN architecture')
    args = parser.parse_args()
    if args.model_name == "base_model":
        cfg = json.load(open("../config/base_model.json", 'r'))
    else:
        raise ValueError("The specified model name is invalid.")

    return args, cfg