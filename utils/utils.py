import argparse
import json
import os
import numpy as np


def upload_args():
    """
    The function loads ALL the parameters passed through COMMAND LINE and also those reported in the configuration file.
    The configuration file is a json.
    The parameters passed through command line have more importance than those reported in the json.
    In case one argument is found in both (json and command line), the json parameter is discarded
    :return:
    """
    parser = argparse.ArgumentParser(description=f'Arguments from command line and json')
    parser.add_argument("--epochs", required=False, type=int, help="Number of epochs")
    parser.add_argument("--input_size", required=False, type=int, help="Input size of a singular time sample")
    parser.add_argument("--hidden_size", required=False, type=int)
    parser.add_argument("--num_layers", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train", required=False, type=bool)
    parser.add_argument("--video", required=False, type=str, help="Video path. Video used for evaluation of results")
    parser.add_argument("--config_file", required=False, type=str, help="Configuration file for additional parameters")
    parser.add_argument("--dataset_path", required=False, type=str, help="Path dataset")
    parser.add_argument("--n_classes", required=False, type=int, help="Number of distinct instruments (classes)")
    args = parser.parse_args()
    args = upload_args_from_json(args)
    print(args)
    return args


def upload_args_from_json(args, file_path=os.path.join("configuration.json")):
    if args is None:
        parser = argparse.ArgumentParser(description=f'Arguments from json')
        args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        # do not override pre-existing arguments, if present.
        # In other terms, the arguments passed through CLI have the priority
        if hasattr(args, option) and getattr(args, option) is not None:
            continue
        if option_value == 'None':
            option_value = None
        if option_value == "True":
            option_value = True
        if option_value == "False":
            option_value = False
        setattr(args, option, option_value)
    return args

