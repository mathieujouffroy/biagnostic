import os
import json
import math
import wandb
import logging
import tensorflow as tf
from train_framework.tf_metrics import *


logger = logging.getLogger(__name__)


class lr_polynomial_decay:
	def __init__(self, epochs, initial_learning_rate, power=1.0):
		self.epochs = epochs
		self.initial_learning_rate = initial_learning_rate
		self.power = power

	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.epochs))) ** self.power
		updated_eta = self.initial_learning_rate * decay
		return float(updated_eta)


class lr_exponential_decay:
	def __init__(self, epochs, initial_learning_rate):
		self.epochs = epochs
		self.initial_learning_rate = initial_learning_rate

	def __call__(self, epoch):
		k = 0.1
		return self.initial_learning_rate * math.exp(-k*epoch)


class lr_time_based_decay:
	def __init__(self, epochs, initial_learning_rate):
		self.epochs = epochs
		self.initial_learning_rate = initial_learning_rate
		self.decay = self.initial_learning_rate/self.epochs

	def __call__(self, epoch, lr):
		return lr * 1 / (1 + self.decay * epoch)


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

    #optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=args.learning_rate, weight_decay=args.decay_rate)
    model.compile(optimizer=optimizer, loss=soft_dice_loss, metrics=args.metrics)

    if args.wandb:
        # monitor the val_loss to save the best model        
        train_curves = wandb.keras.WandbCallback(monitor='val_loss', log_weights=True)
        wandb.define_metric("val_loss", summary="min")
    else:
        train_curves = tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=args.train_dir)
    
    model_dir = f"{args.output_dir}/best_model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_fp =  os.path.join(model_dir, model.name)
    callback_lst = [
        train_curves,
        tf.keras.callbacks.ModelCheckpoint(checkpoint_fp, monitor='val_loss', verbose=0, save_best_only=True),
        #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=args.rate, min_lr=args.min_learnin_rate, verbose=1),
    ]

    logger.info("\n\n")
    logger.info(f"=========== TRAINING MODEL {model.name} ===========")
    logger.info(f"Loss = {soft_dice_loss}")
    logger.info(f"Optimizer = {optimizer}")
    logger.info(f"learning rate = {args.learning_rate}")
    logger.info('\n')

    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set, callbacks=callback_lst)
    
    return model
