import os
import tensorflow as tf
from metrics import *
from model import unet_model_3d
import wandb
import datetime

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

model = unet_model_3d()
