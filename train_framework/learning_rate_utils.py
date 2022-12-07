import math
import wandb
import numpy as np
import tensorflow as tf

class lr_polynomial_decay:
	def __init__(self, epochs, initial_learning_rate, power=1.0):
		self.epochs = epochs
		self.initial_learning_rate = initial_learning_rate
		self.power = power

	def __call__(self, epoch):
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


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """
    def __init__(self,
                 target_lr,
                 total_steps,
                 global_step_init=0,
                 warmup_lr=0.0,
                 warmup_steps=0,
                 hold=0,
                 min_lr=None,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
        Arguments:
            target_lr {float} -- base learning rate.
            total_steps {int} -- total number of training steps.
        Keyword Arguments:
            global_step_init {int} -- initial global step, e.g. from previous checkpoint.
            warmup_lr {float} -- initial learning rate for warm up. (default: {0.0})
            warmup_steps {int} -- number of warmup steps. (default: {0})
            hold {int} -- Optional number of steps to hold base learning rate
                                        before decaying. (default: {0})
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.target_lr = target_lr
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.hold = hold
        self.min_lr = min_lr
        self.learning_rates = []
        self.verbose = verbose
        self.epoch = 0


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch


    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)


    def on_batch_begin(self, batch, logs=None):
        
        lr = self.cosine_decay_with_warmup(
            global_step=self.global_step,
            target_lr=self.target_lr,
            total_steps=self.total_steps,
            warmup_lr=self.warmup_lr,
            warmup_steps=self.warmup_steps,
            hold=self.hold)
        
        if self.min_lr is not None:
            if lr < self.min_lr:
                lr = self.min_lr
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose == 1:
            print(f"\nBatch {self.global_step + 1}: setting learning rate to {lr}")


    def cosine_decay_with_warmup(self, global_step,
                                 target_lr,
                                 total_steps,
                                 warmup_lr=0.0,
                                 warmup_steps=0,
                                 hold=0):

        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')

         # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(
                np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold))
            )

        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        if warmup_steps > 0:
            if target_lr < warmup_lr:
                raise ValueError('target_lr must be larger or equal to '
                                 'warmup_lr.')

            slope = (target_lr - warmup_lr) / warmup_steps
            warmup_rate = slope * global_step + warmup_lr

            learning_rate = np.where(global_step < warmup_steps,
                                    warmup_rate, learning_rate)

        return np.where(global_step > total_steps, 0.0, learning_rate)


class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super(LRLogger, self).__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        wandb.log({"lr": lr}, commit=False)


LR_MAPPINGS = {
    'cosine': WarmUpCosineDecayScheduler,
    'polynomial': lr_polynomial_decay,
    'exponential':lr_exponential_decay,
    'time': lr_time_based_decay,
    'tf_cosine': tf.keras.optimizers.schedules.CosineDecay,
    'tf_exp': tf.keras.optimizers.schedules.ExponentialDecay,
    'tf_invtime': tf.keras.optimizers.schedules.InverseTimeDecay,
}