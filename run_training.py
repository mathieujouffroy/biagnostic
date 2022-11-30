import os
import json
import math
import wandb
import logging
import tensorflow as tf
from train_framework.tf_metrics import *
from train_framework.train import tf_train_model
from train_framework.tf_model import unet_model_3d
from train_framework.utils import set_seed, set_wandb_project_run, parse_args, set_logging
from train_framework.dataloader import BratsDatasetGenerator, TFVolumeDataGenerator

logger = logging.getLogger(__name__)
print(logger)

def main():

    args = parse_args()

    args.train_dir = f"{args.output_dir}train" 

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    set_seed(args)
    
    set_logging(args, 'train')

    # define wandb run and project
    if args.wandb:
        set_wandb_project_run(args, args.m_name)
    
    tf.keras.backend.set_image_data_format("channels_first")

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info(log=True)
    args.len_train = brats_generator.len_train
    args.len_valid = brats_generator.len_val
    #args.class_names = [v for k, v in brats_generator.output_channels.items()]
    args.class_names = list(brats_generator.output_channels.values())

    with open(f"{args.ds_path}split_sets.json", "r") as f:
        set_filenames = json.load(f)

    # Get generators for training and validation sets
    train_generator = TFVolumeDataGenerator(set_filenames['train'], f"{args.ds_path}subvolumes/", batch_size=args.batch_size, dim=args.crop_shape)
    valid_generator = TFVolumeDataGenerator(set_filenames['val'], f"{args.ds_path}subvolumes/", batch_size=args.batch_size, dim=args.crop_shape)

    # Set training parameters
    args.nbr_train_batch = int(math.ceil(args.len_train / args.batch_size))
    # Nbr training steps is [number of batches] x [number of epochs].
    args.n_training_steps = args.nbr_train_batch * args.n_epochs

    logger.info(f"\n  ***** Running training *****\n")
    logger.info(f"  train_set = {train_generator}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Class names = {args.class_names}")
    logger.info(f"  Nbr training examples = {args.len_train}")
    logger.info(f"  Nbr validation examples = {args.len_valid}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Nbr Epochs = {args.n_epochs}")
    logger.info(f"  Nbr of training batch = {args.nbr_train_batch}")
    logger.info(f"  Nbr training steps = {args.n_training_steps}")

    if args.framework == "tf":
        model = unet_model_3d(args.m_name)
        trained_model = tf_train_model(args,  model, train_generator, valid_generator)


    if args.evaluate_during_training:
        args.len_test = brats_generator.len_test
        # load best model & evaluate on test set
        test_generator = TFVolumeDataGenerator(set_filenames['test'], f"{args.ds_path}subvolumes/", batch_size=args.batch_size, dim=args.crop_shape)
        history = trained_model.evaluate(test_generator)
        logger.info(history)
        #print(f"Loss: {loss}")
        #print(f"Average Dice Coefficient on test dataset = {dice_coef:.4f}")
        #print(f"Average Soft Dice Coefficient on test dataset = {soft_dice_coef:.4f}")


    if args.wandb:
        wandb.run.finish()

if __name__ == "__main__":
    main()