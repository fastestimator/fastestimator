import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping_keras
from tensorflow.keras.callbacks import LearningRateScheduler as LearningRateScheduler_keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau_keras
from tensorflow.keras.callbacks import TensorBoard

from fastestimator.estimator.trace import TrainLogger

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
    def __init__(self, pipeline, network, epochs, steps_per_epoch=None, validation_steps=None, traces=None, log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.log_steps = log_steps
        self.rank = 0
        self.local_rank = 0
        self.num_process = 1
        self.num_local_process = 1
        self.traces = traces
        self.do_eval = False
        self.inputs = None

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

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.pipeline.num_examples["train"]//(self.pipeline.batch_size * self.num_process)
        if self.validation_steps is None and self.pipeline.num_examples["eval"] > 0:
            self.validation_steps = self.pipeline.num_examples["eval"]//self.pipeline.batch_size
        self.training_fn = lambda: self.pipeline._input_source("train", self.steps_per_epoch * self.epochs)
        if self.pipeline.num_examples["eval"] > 0 and self.rank ==0:
            self.validation_fn = lambda: self.pipeline._input_source("eval", self.validation_steps)
            self.do_eval = True
        self._add_traces()

    def _add_traces(self):
        self.traces.insert(0, TrainLogger(log_steps=self.log_steps, num_process=self.num_process))

    def train(self):
        self._run_traces_begin(mode="train")
        for train_step, batch in enumerate(self.training_fn()):
            if train_step % self.steps_per_epoch == 0:
                self.epoch = train_step // self.steps_per_epoch
                self._run_traces_on_epoch_begin(mode="train", logs={"epoch": self.epoch})
            self._run_traces_on_batch_begin(mode="train", logs= {"epoch": self.epoch, "step": train_step, "size": self.pipeline.batch_size})
            prediction, loss = self.forward_step(batch, mode="train", epoch=self.epoch)
            self._run_traces_on_batch_end(mode="train", logs= {"epoch": self.epoch, "step": train_step, "size": self.pipeline.batch_size, "batch": batch, "prediction": prediction, "loss": loss})
            # self._run_traces_on_batch_end(mode="train", logs= {"epoch": self.epoch, "step": train_step, "size": self.pipeline.batch_size, "batch": batch, "loss": loss})
            if (train_step + 1) % self.steps_per_epoch == 0:
                self._run_traces_on_epoch_end(mode="train", logs={"epoch": self.epoch})
                if self.do_eval:
                    self.val()
        self._run_traces_end(mode="train")
        print("FastEstimator: training finished!")

    def val(self):
        self._run_traces_begin(mode="eval")
        self._run_traces_on_epoch_begin(mode="eval", logs={"epoch": self.epoch})
        for eval_step, batch in enumerate(self.validation_fn()):
            self._run_traces_on_batch_begin(mode="eval", logs= {"epoch": self.epoch, "step": eval_step, "size": self.pipeline.batch_size})
            prediction, loss = self.forward_step(batch, mode="eval", epoch=self.epoch)
            self._run_traces_on_batch_end(mode="eval", logs= {"epoch": self.epoch, "step": eval_step, "size": self.pipeline.batch_size, "batch": batch, "prediction": prediction, "loss": loss})
            # self._run_traces_on_batch_end(mode="eval", logs= {"epoch": self.epoch, "step": eval_step, "size": self.pipeline.batch_size, "batch": batch, "loss": loss})
        self._run_traces_on_epoch_end(mode="eval", logs={"epoch": self.epoch, "loss": np.mean(np.array(self.losses), axis=0)})
        self._run_traces_end(mode="eval")

    def _run_traces_begin(self, mode):
        for trace in self.traces:
            trace.begin(mode)

    def _run_traces_on_epoch_begin(self, mode, logs):
        self.losses = []
        for trace in self.traces:
            trace.on_epoch_begin(mode, logs)

    def _run_traces_on_batch_begin(self, mode, logs):
        for trace in self.traces:
            trace.on_batch_begin(mode, logs)

    def _run_traces_on_batch_end(self, mode, logs):
        self.losses.append(logs["loss"])
        for trace in self.traces:
            trace.on_batch_end(mode, logs)

    def _run_traces_on_epoch_end(self, mode, logs):
        output_list = []
        for trace in self.traces:
            metric_output = trace.on_epoch_end(mode, logs)
            if mode == "eval" and metric_output is not None:
                trace_name = type(trace).__name__
                output_list.append((trace_name, metric_output))
        if mode == "eval":
            self._print_eval_message(output_list)

    @staticmethod
    def _format_log_message(message, metric_name, metric_value):
        if not isinstance(metric_value, np.ndarray):
            log_message = "{}{}: {}; ".format(message, metric_name, str(metric_value))
        else:
            log_message = "{}\n{}:\n{};\n".format(message, metric_name, np.array2string(metric_value, separator=','))
        return log_message

    def _print_eval_message(self, output_list):
        step = self.steps_per_epoch * (self.epoch + 1)
        log_message = "FastEstimator-Eval: step: {}; ".format(step)
        for metric_name, metric_result in output_list:
            if isinstance(metric_result, dict):
                for key in metric_result.keys():
                    log_message = self._format_log_message(log_message, key, metric_result[key])
            else:
                log_message = self._format_log_message(log_message, metric_name, metric_result)
        print(log_message)

    def _run_traces_end(self, mode):
        for trace in self.traces:
            trace.end(mode)

    @tf.function
    def forward_step(self, batch, mode, epoch):
        num_model = len(self.network.model_list)
        losses = ()
        if mode == "train":
            with tf.GradientTape(persistent=True) as tape:
                prediction = self.network.forward(batch, mode, epoch)
                for idx in range(num_model):
                    losses += self.network.model_list[idx].loss.calculate_loss(batch, prediction),
            for idx in range(num_model):
                gradients = tape.gradient(losses[idx], self.network.model_list[idx].trainable_variables)
                self.network.model_list[idx].optimizer.apply_gradients(zip(gradients,self.network.model_list[idx].trainable_variables))
            del tape
        else:
            prediction = self.network.forward(batch, mode, epoch)
            for idx in range(num_model):
                losses += self.network.model_list[idx].loss.calculate_loss(batch, prediction),
        return prediction, losses