import os
import tensorflow as tf
import wandb
import datetime
import json
import math
from metrics import *
from dataloader import BratsDatasetGenerator, TFVolumeDataGenerator
from model import unet_model_3d
from utils import set_seed, set_wandb_project_run, parse_args

def tf_train_model(args, m_name, model, train_set, valid_set):
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
                    sensitivity, specificity])

    checkpoint = K.callbacks.ModelCheckpoint("3d_unet_brats",
                                         verbose=1,
                                         save_best_only=True)

    callback_lst = [checkpoint]
    logs_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    if args.wandb:
        # monitor the val_loss to save the best model
        wandb_callback = wandb.keras.WandbCallback(monitor='val_loss',log_weights=True)
        callback_lst.append(wandb_callback)
        wandb.define_metric("val_loss", summary="min")
    else:
        callback_lst.append(tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=logs_dir))

    print("\n\n")
    print(f"  =========== TRAINING MODEL {m_name} ===========")
    print(f"  Loss = {args.loss}")
    print(f"  Optimizer = {optimizer}")
    print(f"  learning rate = {args.learning_rate}")
    print('\n')

    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set, callbacks=callback_lst)
    model.save_weights('best_pretrained.hdf5')

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

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info()
    args.len_train = brats_generator.len_train
    args.len_valid = brats_generator.len_valid
    args.len_test = brats_generator.len_test
    args.class_names = [v for k, v in brats_generator.output_channels().items()]
    print(f"  Class names = {args.class_names}")

    with open('../resources/BRATS_ds/config.json', "r")  as f:
        config = json.load(f)

    # Get generators for training and validation sets
    train_generator = TFVolumeDataGenerator(config["Train"]['files'], "../resources/BRATS_ds/Train/", batch_size=3, dim=(160, 160, 32), verbose=1)
    valid_generator = TFVolumeDataGenerator(config["Validation"]['files'], "../resources/BRATS_ds/Validation/", batch_size=3, dim=(160, 160, 32), verbose=1)

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


    model = unet_model_3d()
    m_name = "Unet3D"

    # define wandb run and project
    if args.wandb:
        set_wandb_project_run(args, m_name)

    trained_model = tf_train_model(args, m_name, model, train_generator, valid_generator)

    if args.evaluate_during_training:
        ds_test = brats_generator.get_test_set()
        loss, dice_coef, soft_dice_coef = trained_model.evaluate(ds_test)
        print(f"Loss: {loss}")
        print(f"Average Dice Coefficient on test dataset = {dice_coef:.4f}")
        print(f"Average Soft Dice Coefficient on test dataset = {soft_dice_coef:.4f}")


    if args.wandb:
        wandb.run.finish()

if __name__ == "__main__":
    main()
