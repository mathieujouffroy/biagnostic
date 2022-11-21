import os
import yaml
import argparse
import random
import wandb
import numpy as np
import tensorflow as tf


class YamlNamespace(argparse.Namespace):
    """Namespace from a nested dict returned by yaml.load()"""

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x)
                        if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b)
                        if isinstance(b, dict) else b)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def wandb_cfg(args, n_training_steps):
    # SETUP WANDB
    config_dict = {
        "dataset": args.dataset,
        "nbr_train_epochs": args.n_epochs,
        "nbr_classes": args.n_classes,
        "len_train": args.len_train,
        "len_valid": args.len_valid,
        "len_test": args.len_test,
        "batch_size": args.batch_size,
        "nbr_train_batch": args.nbr_train_batch,
        "n_train_steps": n_training_steps,

        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
    }

    return config_dict


def set_wandb_project_run(args, run_name):
    """ Initialize wandb directory to keep track of our models. """

    project_name = args.output_dir
    cfg = wandb_cfg(args, args.n_training_steps)
    run = wandb.init(project=project_name,
                     job_type="train", name=run_name, config=cfg, reinit=True)
    assert run is wandb.run


def parse_args():
    """ Parse training paremeters from config YAML file. """

    parser = argparse.ArgumentParser(
        description='Train a model for plant disease classification.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="The YAML config file")
    cli_args = parser.parse_args()

    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)

    return config
