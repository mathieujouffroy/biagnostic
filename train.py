import os
import tensorflow as tf
from metrics import *
from model import unet_model_3d
import wandb
import datetime
from utils import set_logging, set_seed, set_wandb_project_run

logger = logging.getLogger(__name__)

def train(args, model, ds_train, ds_val):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=soft_dice_loss,
            metrics=[dice_coefficient, soft_dice_coefficient, precision,
                    sensitivity, specificity])

    checkpoint = K.callbacks.ModelCheckpoint("3d_unet_brats",
                                         verbose=1,
                                         save_best_only=True)

    callback_lst = [checkpoint]
    logs_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if args.wandb:
        # monitor the val_loss to save the best model
        wandb_callback = wandb.keras.WandbCallback(monitor='val_loss',log_weights=True)
        callback_lst.append(wandb_callback)
        wandb.define_metric("val_loss", summary="min")
    else:
        callback_lst.append(tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=logs_dir))

    model.fit(ds_train, epochs=args.n_epochs, validation_data=ds_val, callbacks=callback_lst)
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

    # set logging
    set_logging(args)
    # set seed
    set_seed(args)

    model = unet_model_3d()
    args.class_names = [str(v) for k, v in args.id2label.items()]
    logger.info(f"  Class names = {args.class_names}")

    # Set training parameters
    args.nbr_train_batch = int(math.ceil(args.len_train / args.batch_size))
    # Nbr training steps is [number of batches] x [number of epochs].
    args.n_training_steps = args.nbr_train_batch * args.n_epochs

    logger.info(f"  ---- Training Parameters ----\n\n{args}\n\n")
    logger.info(f"  ***** Running training *****")
    logger.info(f"  train_set = {train_set}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Nbr training examples = {args.len_train}")
    logger.info(f"  Nbr validation examples = {args.len_valid}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Nbr Epochs = {args.n_epochs}")
    logger.info(f"  Nbr of training batch = {args.nbr_train_batch}")
    logger.info(f"  Nbr training steps = {args.n_training_steps}")
    logger.info(f"  Class weights = {class_weights}")

    # define wandb run and project
    if args.wandb:
        set_wandb_project_run(args, m_name)

    trained_model = train_model(
        args, m_name, model, train_set, valid_set, class_weights)

    if args.eval_during_training:
        if args.transformer:
            args.input_shape = (3, 224, 224)
            test_set = load_from_disk(f'{ds_path}/test')
            args.len_test = test_set.num_rows
            data_collator = DefaultDataCollator(return_tensors="tf")
            test_set = test_set.to_tf_dataset(
                columns=['pixel_values'],
                label_cols=["labels"],
                shuffle=True,
                batch_size=32,
                collate_fn=data_collator)
        else:
            X_test, y_test = load_split_hdf5(ds_path, 'test')
            args.len_test = len(X_test)
            test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            del X_test, y_test
            gc.collect()
        test_set = prep_ds_input(args, test_set, args.len_test, img_size)
        logger.info(f"\n  ***** Evaluating on Test set *****")
        compute_training_metrics(args, trained_model, m_name, test_set)
    
    if args.wandb:
        wandb.run.finish()

if __name__ == "__main__":
    main()