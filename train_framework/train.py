import os
import json
import math
import wandb
import logging
import tensorflow as tf
from train_framework.tf_metrics import *
from train_framework.learning_rate_utils import *

logger = logging.getLogger(__name__)

OPTIMIZER_MAPPINGS = {
    'Adam': tf.keras.optimizers.Adam,
    'AdamW': tf.keras.optimizers.experimental.AdamW,
    'SGD': tf.keras.optimizers.SGD,
}

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

    model_dir = f"{args.output_dir}/best_model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = OPTIMIZER_MAPPINGS[args.optimizer]
    if args.optimizer != 'SGD':
        if args.weight_decay:
            optimizer = optimizer(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer(learning_rate=args.learning_rate)
    else:
        optimizer = optimizer(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=soft_dice_loss, metrics=args.metrics)


    if args.lr_scheduler in ['polynomial', 'exponential', 'time']:
        scheduler = LR_MAPPINGS[args.lr_scheduler]
        scheduler = scheduler(args.n_epochs, args.learning_rate)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    elif args.lr_scheduler in ['tf_cosine', 'tf_exp', 'tf_invtime']:
        scheduler = LR_MAPPINGS[args.lr_scheduler]
        if args.lr_scheduler == 'tf_cosine':
            scheduler = scheduler(args.learning_rate, args.n_training_steps)
        else:
            scheduler = scheduler(args.learning_rate, args.n_training_steps, args.lr_decay_rate)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    
    elif args.lr_scheduler == 'cosine':    
        warmup_steps = int(args.warmup_epoch * args.len_train / args.batch_size)
        lr_callback = WarmUpCosineDecayScheduler(target_lr=args.learning_rate, total_steps=args.n_training_steps, warmup_lr=0.0, 
                        warmup_steps=warmup_steps, hold=0, min_lr=args.min_learnin_rate)
                        #warmup_steps=warmup_steps, hold=(args.len_train / args.batch_size), min_lr=args.min_learnin_rate)

    elif args.lr_scheduler == 'plateau_reduce':
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=args.lr_decay_rate, min_lr=args.min_learnin_rate, verbose=1),


    if args.wandb:
        # monitor the val_loss to save the best model        
        train_curves = wandb.keras.WandbCallback(monitor=args.track_metric, log_weights=True)
        wandb.define_metric(args.track_metric, summary=args.track_obj)
    else:
        train_curves = tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=args.train_dir)


    checkpoint_fp =  os.path.join(model_dir, model.name)
    callback_lst = [
        train_curves,
        tf.keras.callbacks.ModelCheckpoint(checkpoint_fp, monitor=args.track_metric, verbose=0, save_best_only=True),
        LRLogger(optimizer),
    ]
    if args.lr_scheduler:
        callback_lst.append(lr_callback)

    logger.info("\n\n")
    logger.info(f"=========== TRAINING MODEL {model.name} ===========")
    logger.info(f"Loss = {soft_dice_loss}")
    logger.info(f"Optimizer = {optimizer}")
    logger.info(f"learning rate = {args.learning_rate}")
    logger.info('\n')

    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set, callbacks=callback_lst)
    
    return model


