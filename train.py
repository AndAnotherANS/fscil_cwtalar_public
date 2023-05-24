import argparse
import tomllib

import wandb

from src.fscil_trainer import FSCILTrainer
from utils import *

class Namespace:
    def __init__(self, d):
        self.__dict__.update(d)


def parse_config(path):
    with open(path, "rb") as file:
        config_dict = tomllib.load(file)
    return Namespace(config_dict)

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

    args = parse_config(config_path)

    pprint(vars(args))

    set_up_wandb(args)

    args.num_gpu = set_gpu(args)
    args.device = "cuda" if args.num_gpu != 0 else "cpu"
    trainer = FSCILTrainer(args)
    trainer.train()