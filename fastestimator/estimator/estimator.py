from fastestimator.estimator.callbacks import OutputLogger, LearningRateUpdater, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping_keras
from tensorflow.keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau_keras
from tensorflow.keras.callbacks import LearningRateScheduler as LearningRateScheduler_keras
from tensorflow.keras import backend as K
import tensorflow as tf
import logging
import sys
import os

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
        if self.num_process > 1:
            self._horovod_setup()
        self._prepare_network()
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
        self.pipeline.input_names = self.network.input_names
        self.pipeline.output_names = self.network.output_names
        self.pipeline._prepare(self.inputs)
        self.training_fn = lambda: self.pipeline._input_source("train")
        if self.pipeline.num_examples["eval"] > 0 and self.rank ==0:
            self.validation_fn = lambda: self.pipeline._input_source("eval")

    def _prepare_network(self):
        if hasattr(self.network.model, "build_later"):
            self.network.model = self.network.model.build_later()
        self.network._get_inout_list()
        self.network.model.compile(optimizer=self.network.optimizer, loss=self.network.loss, metrics=self.network.metrics, loss_weights=self.network.loss_weights)
        self.network.init_lr = K.get_value(self.network.optimizer.lr)
        if self.rank == 0:
            print("FastEstimator: Model artifact will be saved in %s" % self.network.model_dir)

    def _prepare_estimator(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.pipeline.num_examples["train"]//(self.pipeline.batch_size * self.num_process)
        if self.validation_steps is None and self.pipeline.num_examples["eval"] > 0:
            self.validation_steps = self.pipeline.num_examples["eval"]//self.pipeline.batch_size
        self._add_callbacks()

    def _add_callbacks(self):
        lr_update_obj = LearningRateUpdater(init_lr= self.network.init_lr)
        model_saving = False
        for callback_obj in self.callbacks:
            callback_obj.lr_update_obj = lr_update_obj
            if isinstance(callback_obj, (EarlyStopping_keras, ReduceLROnPlateau_keras, LearningRateScheduler_keras)):
                raise ValueError("please use FastEstimator version of following callback: %s" % str(callback_obj))
            if isinstance(callback_obj, ModelCheckpoint):
                if self.rank == 0:
                    model_saving = True
                else:
                    self.callbacks.remove(callback_obj)
            if isinstance(callback_obj, TensorBoard) and self.rank != 0:
                self.callbacks.remove(callback_obj)
            if hasattr(callback_obj, "num_process"):
                callback_obj.num_process = self.num_process
            if hasattr(callback_obj, "rank"):
                callback_obj.rank = self.rank
            if isinstance(callback_obj, LearningRateScheduler):
                callback_obj.schedule.epochs = self.epochs
                callback_obj.schedule.steps_per_epoch = self.steps_per_epoch
        if not model_saving and self.rank == 0:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self.network.model_dir, "BestModel.h5"), save_best_only=self.pipeline.num_examples["eval"] > 0))
        self.callbacks.append(lr_update_obj)
        if self.rank == 0:
            self.callbacks.append(OutputLogger(batch_size=self.pipeline.batch_size, validation=self.pipeline.num_examples["eval"] > 0, log_steps=self.log_steps, num_process=self.num_process))
        
    def train(self):
        if self.pipeline.num_examples["eval"] > 0 and self.rank ==0:
            self.network.model.fit(self.training_fn(),
                                    validation_data= self.validation_fn(),
                                    validation_steps= self.validation_steps,
                                    steps_per_epoch= self.steps_per_epoch,
                                    epochs= self.epochs,
                                    callbacks= self.callbacks,
                                    verbose= 0)
        else:
            self.network.model.fit(self.training_fn(),
                                    steps_per_epoch= self.steps_per_epoch,
                                    epochs= self.epochs,
                                    callbacks= self.callbacks,
                                    verbose= 0)

    def _horovod_setup(self):
        import horovod.tensorflow.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
        self.network.optimizer = hvd.DistributedOptimizer(self.network.optimizer)
        self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.num_local_process = hvd.local_size()
