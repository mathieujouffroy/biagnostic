import os
import json
import logging
import tensorflow as tf
from train_framework.tf_metrics import *
from train_framework.utils import set_seed, parse_args, set_logging
from train_framework.dataloader import BratsDatasetGenerator, TFVolumeDataGenerator

logger = logging.getLogger(__name__)
print(logger)

def main():

    args = parse_args()

    if not os.path.exists(f"{args.output_dir}infer"):
        os.makedirs(f"{args.output_dir}infer")

    set_seed(args)

    set_logging(args, 'infer')

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info(log=True)
    args.len_test = brats_generator.len_test
    #args.class_names = [v for k, v in brats_generator.output_channels.items()]
    args.class_names = list(brats_generator.output_channels.values())
    
    with open(f"{args.ds_path}split_sets.json", "r") as f:
        set_filenames = json.load(f)

    test_generator = TFVolumeDataGenerator(set_filenames['test'], f"{args.ds_path}subvolumes/", batch_size=args.batch_size, dim=args.crop_shape)

    logger.info(f"\n\n  ***** Running Inference *****\n")
    logger.info(f"  test_set = {test_generator}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Class names = {args.class_names}")
    logger.info(f"  Nbr test examples = {args.len_test}")
    logger.info(f"  Batch size = {args.batch_size}\n\n")

    custom_metrics = {"soft_dice_loss":soft_dice_loss, "dice_coefficient":dice_coefficient,
                    "soft_dice_coefficient":soft_dice_coefficient,
                    "precision":precision, "sensitivity":sensitivity, "specificity":specificity, 
                    "iou":iou}

    trained_model = tf.keras.models.load_model(f"{args.output_dir}/best_model/{args.m_name}", custom_objects=custom_metrics)

    history = trained_model.evaluate(test_generator)
    logger.info(history)

if __name__ == "__main__":
    main()