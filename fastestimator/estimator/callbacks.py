from tensorflow.keras import backend as K
import logging
import numpy as np
import tensorflow as tf
import time

class OutputLogger(tf.keras.callbacks.Callback):
    """
    Keras callback for logging the output

    Args:
        batch_size: Size of the training batch
        log_steps: Number of steps at which to output the logs
        validation: Boolean representing whether or not to output validation information
    """
    def __init__(self, batch_size, log_steps=100, validation=True, num_process=1):
        self.batch_size = batch_size
        self.log_steps = log_steps
        self.global_step = -1
        self.times = []
        self.epochs_since_best = 0
        self.best_loss = None
        self.validation = validation
        self.num_process = num_process

    def on_batch_begin(self, batch, logs=None):
        self.global_step += 1
        self.time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        self.times.append(time.time() - self.time_start)
        if self.global_step % self.log_steps == 0:
            current_lr = K.get_value(self.model.optimizer.lr)
            if self.global_step == 0:
                example_per_second = 0.0
            else:
                if len(self.times) >= 10:
                    average_time = np.mean(self.times[4:-5])
                else:
                    average_time = np.mean(self.times)
                example_per_second = self.batch_size / average_time
            loss = logs["loss"]
            print("FastEstimator-Train:step: %d; train_loss: %f; lr: %f; example/sec: %f;" %(self.global_step, loss, current_lr, example_per_second*self.num_process))
            self.times = []

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            current_eval_loss = logs["val_loss"]
            if self.best_loss == None or current_eval_loss < self.best_loss:
                self.best_loss = current_eval_loss
                self.epochs_since_best = 0
            else:
                self.epochs_since_best += 1
            log_message = "step: %d; val_loss: %f; min_val_loss: %f; since_best: %d; " % (self.global_step+1, current_eval_loss, self.best_loss, self.epochs_since_best)
            for key in logs.keys():
                if "val_" in key and key != "val_loss":
                    eval_value = logs[key]
                    log_message = log_message + key + ": " + str(eval_value) + "; "
            print("FastEstimator-Eval:" + log_message)


class LearningRateUpdater(tf.keras.callbacks.Callback):
    """
    Keras callback to update the learning rate

    Args:
        init_lr: initial learning rate
    """
    def __init__(self, init_lr):
        self.init_lr = init_lr
        self.needs_update = False
        self.lr_schedule_factor = 1.0
        self.reducelr_factor = 1.0
    
    def on_batch_begin(self, batch, logs=None):
        if self.needs_update:
            K.set_value(self.model.optimizer.lr, self.init_lr * self.lr_schedule_factor * self.reducelr_factor)
    
    def on_batch_end(self, batch, logs={}):
        if self.needs_update:
            self.needs_update = False


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Keras callback for the learning rate scheduler

    Args:
        schedule: Schedule object to passed to the scheduler
    """  
    def __init__(self, schedule):
        self.schedule = schedule
        self.lr_update_obj = None
        self.global_steps = 0

    def on_batch_begin(self, batch, logs=None):
        if self.schedule.mode == "global_steps":
            self.lr_update_obj.lr_schedule_factor = self.schedule.lr_schedule_fn(self.global_steps)
            self.global_steps += 1
            self.lr_update_obj.needs_update = True

    def on_epoch_begin(self, epoch, logs=None):
        if self.schedule.mode == "epoch":
            self.lr_update_obj.lr_schedule_factor = self.schedule.lr_schedule_fn(epoch)
            self.lr_update_obj.needs_update = True


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Keras callback for the reduce learning rate on pleateau
    
    Args:
        monitor: Metric to be monitored
        factor: Factor by which to reduce learning rate
        patience: Number of epochs to wait before reducing LR
        verbose: Whether or not to output verbose logs
        mode: Learning rate reduction mode
        min_delta: Minimum significant difference
        cooldown:
        **kwargs:
    """
    def __init__(self,
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=0,
                mode='auto',
                min_delta=1e-4,
                cooldown=0,
                **kwargs):
        super(ReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.lr_update_obj = None
        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.num_process = 1
        self.rank = 0
        self._reset()

    def _reset(self):
        """
        Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        if self.rank == 0:
            logs = logs or {}
            logs['lr'] = self.lr_update_obj.init_lr * self.lr_update_obj.lr_schedule_factor * self.lr_update_obj.reducelr_factor
            current = logs.get(self.monitor)
            if current is None:
                logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))
            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.lr_update_obj.reducelr_factor = np.float32(self.lr_update_obj.reducelr_factor * self.factor)
                        self.lr_update_obj.needs_update = True
                        print('FastEstimator: ReduceLROnPlateau reducing learning rate to %s.' % (self.lr_update_obj.init_lr * self.lr_update_obj.lr_schedule_factor * self.lr_update_obj.reducelr_factor))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        if self.num_process > 1:
            import horovod.tensorflow.keras as hvd
            self.lr_update_obj.reducelr_factor = hvd.broadcast(self.lr_update_obj.reducelr_factor, 0)
            self.lr_update_obj.needs_update = hvd.broadcast(self.lr_update_obj.needs_update, 0)

    def in_cooldown(self):
        return self.cooldown_counter > 0

class EarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None):
        super(EarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.rank = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if self.rank == 0:
            current = logs.get(self.monitor)
            if current is None:
                logging.warning('Early stopping conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("FastEstimator: EarlyStopping stop training")
        if self.num_process > 1:
            import horovod.tensorflow.keras as hvd
            self.model.stop_training = hvd.broadcast(self.model.stop_training, 0)