import argparse
import copy
import itertools
import tomllib
from sklearn.model_selection import ParameterGrid

import wandb

from src.fscil_trainer import FSCILTrainer
from utils import *

class Namespace:
    def __init__(self, d):
        self.__dict__.update(d)


def parse_config(path):
    with open(path, "rb") as file:
        config_dict = tomllib.load(file)
    for key, val in config_dict.items():
        if not isinstance(val, list):
            config_dict[key] = [val]
    params = ParameterGrid(config_dict)
    return [Namespace(p) for p in params]

def set_up_wandb(args):
    if hasattr(args, "wandb_entity") and hasattr(args, "wandb_project"):
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,

            # track hyperparameters and run metadata
            config=vars(args)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file path")
    config_path = parser.parse_args().config

    args_list = parse_config(config_path)
    set_up_wandb(args_list[0])

    for args in args_list:

        args.num_gpu = set_gpu(args)
        args.device = "cuda" if args.num_gpu != 0 else "cpu"
        trainer = FSCILTrainer(args)
        trainer.train()
