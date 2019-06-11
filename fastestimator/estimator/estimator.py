from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping_keras
from tensorflow.keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau_keras
from tensorflow.keras.callbacks import LearningRateScheduler as LearningRateScheduler_keras
from tensorflow.keras import backend as K
import tensorflow as tf
import logging
import time
import sys
import os
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Estimator:
    """
    ``Estimator`` class compiles all the components necessary to train a model.

    Args:
        pipeline: Object of the Pipeline class that consists of data parameters.
        network: Object of the Network class that consists of the model definition and parameters.
        epochs: Total number of training epochs
        steps_per_epoch: The number batches in one epoch of training,
            if None, it will be automatically calculated. Evaluation is performed at the end of every epoch.
            (default: ``None``)
        validation_steps: Number of batches to be used for validation
        callbacks: List of callbacks object in tf.keras. (default: ``[]``)
        log_steps: Number of steps after which training logs will be displayed periodically.
    """
    def __init__(self, pipeline, network, epochs, steps_per_epoch=None, validation_steps=None, callbacks=[], log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self.log_steps = log_steps
        self.rank = 0
        self.local_rank = 0
        self.num_process = 1
        self.num_local_process = 1
    
    def fit(self, inputs=None):
        """
        Function to perform training on the estimator
        
        Args:
            inputs: Path to input data

        Returns:
            None
        """
        self.inputs = inputs
        self._prepare_pipeline()
        self._prepare_estimator()
        self.train()

    def _prepare_pipeline(self):
        if self.inputs is None and self.pipeline.train_data is None:
            raise ValueError("Must specify the data path when using existing tfrecords")
        for feature in self.pipeline.feature_name:
            if "/" in feature:
                raise ValueError("Feature name should not contain '/'")
        self.pipeline.num_process = self.num_process
        self.pipeline.num_local_process = self.num_local_process
        self.pipeline.rank = self.rank
        self.pipeline.local_rank = self.local_rank
        self.pipeline._prepare(self.inputs)
        self.training_fn = lambda: self.pipeline._input_source("train")
        if self.pipeline.num_examples["eval"] > 0 and self.rank ==0:
            self.validation_fn = lambda: self.pipeline._input_source("eval")

    def _prepare_estimator(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.pipeline.num_examples["train"]//(self.pipeline.batch_size * self.num_process)
        if self.validation_steps is None and self.pipeline.num_examples["eval"] > 0:
            self.validation_steps = self.pipeline.num_examples["eval"]//self.pipeline.batch_size

    def train(self):
        def _reshape(data, shape):
            return np.reshape(data, shape)

        def _minmax(data):
            epsilon = 1e-7
            data = data.astype(np.float32)
            data = data - np.reshape(np.min(data, axis=(1,2,3)), (32,1,1,1))
            data = data / (np.reshape(np.max(data, axis=(1,2,3)), (32,1,1,1)) + epsilon)
            return data
        start = time.time()
        global_step = 0
        for feature in self.training_fn():
            feature["x"] = _reshape(feature["x"], shape=(32, 28, 28, 1))
            feature["x"] = _minmax(feature["x"])
            loss = self.train_step(feature)
            global_step += 1
            if global_step % 100 == 0 and global_step > 0:
                time_elapse = time.time() - start
                example_per_sec = 100 * self.pipeline.batch_size / time_elapse
                if isinstance(loss, tuple):
                    loss = [x.numpy() for x in list(loss)]
                else:
                    loss = loss.numpy()
                print("step: %d, loss is %s, example/sec is %f" % (global_step, str(loss), example_per_sec))
                # print("step: %d, example/sec is %f" % (global_step, example_per_sec))
                start = time.time()
    
    @tf.function
    def train_step(self, features):
        loss = self.network.train_op(features)
        return loss