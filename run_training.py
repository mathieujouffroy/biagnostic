import os
import json
import math
import wandb
import logging
import torch
import yaml
import tensorflow as tf
from train_framework.tf_metrics import *
#from train_framework.pt_metrics import *
from train_framework.train import tf_train_model
from train_framework.tf_model import Unet3D, AttentionUnet3D
from train_framework.utils import set_seed, set_wandb_project_run, parse_args, set_logging
from train_framework.dataloader import BratsDatasetGenerator, TFVolumeDataGenerator, VolumeDataset

logger = logging.getLogger(__name__)
print(logger)

def run_model(args):
    print("running")
    args.train_dir = f"{args.output_dir}/train" 
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    set_seed(args)
    set_logging(args, 'train')

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info(log=True)
    args.len_train = brats_generator.len_train
    args.len_valid = brats_generator.len_val
    args.len_test = brats_generator.len_test
    args.class_names = list(brats_generator.output_channels.values())
    args.crop_dim = (args.crop_shape[0], args.crop_shape[1], args.crop_shape[2], args.n_channels)
    
    with open(f"{args.ds_path}split_sets.json", "r") as f:
        set_filenames = json.load(f)

    # Set training parameters
    args.nbr_train_batch = args.len_train // args.batch_size
    args.n_training_steps = args.nbr_train_batch * args.n_epochs


    # Get generators for training and validation sets
    if args.framework == 'tf':
        #train_generator = TFVolumeDataGenerator(
        #                    set_filenames['train'],
        #                    f"{args.ds_path}subvolumes/", 
        #                    batch_size=args.batch_size, 
        #                    dim=args.crop_shape,
        #                    n_channels=args.n_channels,
        #                    n_classes=args.n_classes,
        #                    shuffle=True,
        #                    augmentation=args.augmentation
        #                )
        #valid_generator = TFVolumeDataGenerator(
        #                    set_filenames['val'], 
        #                    f"{args.ds_path}subvolumes/",
        #                    batch_size=args.batch_size,
        #                    dim=args.crop_shape,
        #                    n_channels=args.n_channels,
        #                    n_classes=args.n_classes,
        #                    shuffle=True
        #                )
        #for elem in train_generator.__getitem__(1):
        #    logger.info(f"batch shape is {elem.shape}, type is {elem.dtype}")
        train_generator = brats_generator.get_train()
        valid_generator = brats_generator.get_valid()
        for elem, label in train_generator.take(1):
           logger.info(f"elem shape is {elem.shape} type: {elem.dtype}, {elem.shape.ndims}")
           logger.info(f"label shape is {label.shape} type: {label.dtype}, {label.shape.ndims}")

    else:
        train_set = VolumeDataset(set_filenames['train'], f"{args.ds_path}subvolumes/", dim=args.crop_shape, transform=args.augmentation)
        val_set = VolumeDataset(set_filenames['val'], f"{args.ds_path}subvolumes/", dim=args.crop_shape)
        train_generator = torch.utils.data.DataLoader(train_set, {'batch_size': 4,'shuffle': True,'num_workers': 6})
        valid_generator  = torch.utils.data.DataLoader(val_set, {'batch_size': 4,'shuffle': True,'num_workers': 6})
                

    logger.info(f"\n  ***** Running training *****\n")
    logger.info(f"  train_set = {train_generator}")
    logger.info(f"  Nbr of channels = {args.n_channels}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Class names = {args.class_names}")
    logger.info(f"  Input Shape : {(args.crop_shape[0], args.crop_shape[1], args.crop_shape[2], args.n_channels)}")
    logger.info(f"  Nbr training examples = {args.len_train}")
    logger.info(f"  Nbr validation examples = {args.len_valid}")


    # define wandb run and project
    if args.wandb:
        args = set_wandb_project_run(args, args.m_name)
        print(args)


    if args.framework == "tf":
        if args.m_name.split('-')[0] == 'Attention':
            model = AttentionUnet3D(args.m_name, args.crop_dim, 3)
        else:
            model = Unet3D(args.m_name, args.crop_dim, 3)

        model = model.build()
        args.metrics =[dice_loss, dice_coefficient, soft_dice_coefficient, iou_coeff]
        args.loss = LOSS_MAPPINGS[args.loss_name]
        if args.loss_name == 'soft_dice_loss':
            args.metrics.append(dice_loss)
        else:
            args.metrics.append(soft_dice_loss)

        trained_model = tf_train_model(args, model, train_generator, valid_generator)


    if args.evaluate_during_training:
        # load best model & evaluate on test set
        #test_generator = TFVolumeDataGenerator(set_filenames['test'], f"{args.ds_path}subvolumes/", batch_size=args.batch_size, dim=args.crop_shape)
        test_generator = brats_generator.get_test()
        history = trained_model.evaluate(test_generator)
        logger.info(f"  history:{history}")
        loss, dice_coef, soft_dice_coef = history[0], history[2], history[2]
        iou_coef, mean_iou = history[3], history[4]
        
        logger.info(f"Loss on test dataset = {loss}")
        logger.info(f"Average Dice Coefficient on test dataset = {dice_coef:.4f}")
        logger.info(f"Average Soft Dice Coefficient on test dataset = {soft_dice_coef:.4f}")
        logger.info(f"Average IOU Coefficient on test dataset = {iou_coef:.4f}")
        logger.info(f"Average Mean IOU on test dataset = {mean_iou:.4f}")


    if args.wandb:
        wandb.run.finish()

if __name__ == "__main__":
    args = parse_args()
    run_model(args)

## add class weights