import os
import json
import math
import wandb
import datetime
import tensorflow as tf
from tf_metrics import *
from tf_model import unet_model_3d
from utils import set_seed, set_wandb_project_run, parse_args
from dataloader import BratsDatasetGenerator, TFVolumeDataGenerator

def tf_train_model(args, model, train_set, valid_set):
    """
    Compiles and fits the model.
    Parameters:
        args: Argument Parser
        m_name: Model name
        model: Model to train
        train_set(tensorflow.Dataset): training set
        valid_set(tensorflow.Dataset): validation set
    Returns:
        model(tensorflow.Model): trained model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=soft_dice_loss,
            metrics=[dice_coefficient, soft_dice_coefficient, precision,
                    sensitivity, specificity, iou])


    logs_dir = os.path.join(args.output_dir, "train_logs")
    checkpoint_fp =  os.path.join(args.output_dir, f"best_model/{model.name}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_fp, monitor='val_loss', verbose=1, save_best_only=True)

    callback_lst = [checkpoint]
    

    if args.wandb:
        # monitor the val_loss to save the best model        
        report_cb = wandb.keras.WandbCallback(monitor='val_loss', log_weights=True)
        wandb.define_metric("val_loss", summary="min")
    else:
        report_cb = tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=logs_dir)
    
    callback_lst.append(report_cb)

    print("\n\n")
    print(f"=========== TRAINING MODEL {model.name} ===========")
    print(f"Loss = {soft_dice_loss}")
    print(f"Optimizer = {optimizer}")
    print(f"learning rate = {args.learning_rate}")
    print('\n')

    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set, callbacks=callback_lst)
    #model.save_weights(f'{args.output_dir}best_pretrained.hdf5')

    return model


def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # set seed
    set_seed(args)

    #
    tf.keras.backend.set_image_data_format("channels_first")

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info()
    args.len_train = brats_generator.len_train
    args.len_valid = brats_generator.len_val
    args.len_test = brats_generator.len_test
    args.class_names = [v for k, v in brats_generator.output_channels.items()]
    print(f"  Class names = {args.class_names}")

    with open(f"{args.ds_path}split_sets.json", "r") as f:
        set_filenames = json.load(f)

    # Get generators for training and validation sets
    train_generator = TFVolumeDataGenerator(set_filenames['train'], f"{args.ds_path}subvolumes/", batch_size=3, dim=(128, 128, 32))
    valid_generator = TFVolumeDataGenerator(set_filenames['val'], f"{args.ds_path}subvolumes/", batch_size=3, dim=(128, 128, 32))

    # Set training parameters
    args.nbr_train_batch = int(math.ceil(args.len_train / args.batch_size))
    # Nbr training steps is [number of batches] x [number of epochs].
    args.n_training_steps = args.nbr_train_batch * args.n_epochs

    print(f"  ---- Training Parameters ----\n\n{args}\n\n")
    print(f"  ***** Running training *****")
    print(f"  train_set = {train_generator}")
    print(f"  Nbr of class = {args.n_classes}")
    print(f"  Nbr training examples = {args.len_train}")
    print(f"  Nbr validation examples = {args.len_valid}")
    print(f"  Batch size = {args.batch_size}")
    print(f"  Nbr Epochs = {args.n_epochs}")
    print(f"  Nbr of training batch = {args.nbr_train_batch}")
    print(f"  Nbr training steps = {args.n_training_steps}")

    model = unet_model_3d(args.m_name)
    print(model.name)

    # define wandb run and project
    if args.wandb:
        set_wandb_project_run(args, args.m_name)

    trained_model = tf_train_model(args,  model, train_generator, valid_generator)

    if args.evaluate_during_training:
        # load best model & evaluate on test set
        test_generator = TFVolumeDataGenerator(set_filenames['test'], f"{args.ds_path}subvolumes/", batch_size=3, dim=(128, 128, 32))
        # model =
        custom_metrics = {"dice_coefficient":dice_coefficient,
                        "soft_dice_coefficient":soft_dice_coefficient,
                        "precision":precision, "sensitivity":sensitivity, "specificity":specificity, 
                        "iou":iou}

        #trained_model = tf.keras.models.load_model(f"{args.output_dir}/best_model", custom_objects=custom_metrics)
        history = trained_model.evaluate(test_generator)
        print(history)
        #print(f"Loss: {loss}")
        #print(f"Average Dice Coefficient on test dataset = {dice_coef:.4f}")
        #print(f"Average Soft Dice Coefficient on test dataset = {soft_dice_coef:.4f}")


    if args.wandb:
        wandb.run.finish()

if __name__ == "__main__":
    main()
