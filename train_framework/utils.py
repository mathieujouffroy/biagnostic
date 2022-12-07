import os
import yaml
import wandb
import random
import logging
import argparse
import datetime
import numpy as np
import tensorflow as tf


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def set_logging(args, log_type):
    "Defines the file in which we will write our training logs"
    
    date = datetime.datetime.now().strftime("%d:%m-%H:%M")
    if log_type == 'train':
        log_file = os.path.join(f"{args.output_dir}/train", f"{args.m_name}_{date}.log")
    elif log_type == 'infer':
        log_file = os.path.join(f"{args.output_dir}/infer", f"{args.m_name}_{date}.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )


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


def wandb_cfg(args):
    # SETUP WANDB
    config_dict = {
        "dataset": args.ds_path,
        "len_train": args.len_train,
        "len_valid": args.len_valid,
        "len_test": args.len_test,

        "nbr_epochs": args.n_epochs,
        "nbr_classes": args.n_classes,

        "loss": args.loss,
        

        "batch_size": args.batch_size,
        "nbr_train_batch": args.nbr_train_batch,
        "train_steps": args.n_training_steps,
        "learning_rate": args.learning_rate,
        "img_shape": args.crop_shape,
    }
    return config_dict



def set_wandb_project_run(args, run_name):
    """ Initialize wandb directory to keep track of our models. """

    cfg = wandb_cfg(args)
    run = wandb.init(project='brats-3d-segm',
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

