import os
import json
import math
import wandb
import logging
import tensorflow as tf
from train_framework.tf_metrics import *


logger = logging.getLogger(__name__)


def scheduler(epoch, lr):
  if epoch <= 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1*epoch)


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
    #tf.keras.metrics.Precision
    #tf.keras.metrics.IoU, tf.keras.metrics.MeanIoU
    # tf.keras.metrics.OneHotIoU, tf.keras.metrics.OneHotMeanIoU
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=soft_dice_loss,
            #metrics=[dice_coefficient, soft_dice_coefficient, precision,
            #        sensitivity, specificity, iou])
            metrics=[iou, tf.keras.metrics.IoU, tf.keras.metrics.OneHotIoU,
            precision, specificity, tf.keras.metrics.Precision,
            sensitivity, tf.keras.metrics.Recall
            ])


    if args.wandb:
        # monitor the val_loss to save the best model        
        train_curves = wandb.keras.WandbCallback(monitor='val_loss', log_weights=True)
        wandb.define_metric("val_loss", summary="min")
    else:
        train_curves = tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=args.train_dir)
    
    checkpoint_fp =  os.path.join(args.output_dir, f"best_model/{model.name}")
    callback_lst = [
        train_curves,
        tf.keras.callbacks.ModelCheckpoint(checkpoint_fp, monitor='val_loss', verbose=0, save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    ]

    logger.info("\n\n")
    logger.info(f"=========== TRAINING MODEL {model.name} ===========")
    logger.info(f"Loss = {soft_dice_loss}")
    logger.info(f"Optimizer = {optimizer}")
    logger.info(f"learning rate = {args.learning_rate}")
    logger.info('\n')

    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set, callbacks=callback_lst)
    
    return model
