import numpy as np
import tensorflow as tf
from fastestimator.estimator.trace import TrainLogger

class Estimator:
    def __init__(self, pipeline, network, epochs, steps_per_epoch=None, validation_steps=None, traces=None, log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.log_steps = log_steps
        self.traces = traces
        self.num_gpu = 1

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
        self._prepare_network()
        self._prepare_estimator()
        self.train()

    def _prepare_pipeline(self):
            self.pipeline._prepare(inputs = self.inputs)
            assert self.pipeline._preprare_mode(mode="train"), "could not find data in {} for training"
            self.do_eval = self.pipeline._preprare_mode(mode="eval")
            
    def _prepare_network(self):
        self.network._check_ops("train")
        if self.do_eval:
            self.network._check_ops("eval")

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        if self.steps_per_epoch is None:
            self.steps_per_epoch = np.min(self.pipeline.num_examples["train"])//(self.pipeline.batch_size * self.num_gpu)
        if self.validation_steps is None and self.do_eval:
            self.validation_steps = np.min(self.pipeline.num_examples["eval"])//self.pipeline.batch_size
        self.training_fn = lambda: self.pipeline._input_stream("train")
        if self.do_eval:
            self.validation_fn = lambda: self.pipeline._input_stream("eval")
        self._add_traces()

    def _add_traces(self):
        self.traces.insert(0, TrainLogger(log_steps=self.log_steps, num_process=self.num_gpu))

    def train(self):
        self._run_traces_begin(mode="train")
        for train_step, batch in enumerate(self.training_fn().take(self.steps_per_epoch * self.epochs)):
            if train_step % self.steps_per_epoch == 0:
                self.epoch = train_step // self.steps_per_epoch
                self._run_traces_on_epoch_begin(mode="train", logs={"epoch": self.epoch})
            self._run_traces_on_batch_begin(mode="train", logs= {"epoch": self.epoch, "step": train_step, "size": self.pipeline.batch_size})
            prediction, loss = self.forward_step(batch, mode="train", epoch=self.epoch)
            self._run_traces_on_batch_end(mode="train", logs= {"epoch": self.epoch, "step": train_step, "size": self.pipeline.batch_size, "batch": batch, "prediction": prediction, "loss": loss})
            if (train_step + 1) % self.steps_per_epoch == 0:
                self._run_traces_on_epoch_end(mode="train", logs={"epoch": self.epoch})
                if self.do_eval:
                    self.val()
        self._run_traces_end(mode="train")
        print("FastEstimator: training finished!")

    def val(self):
        self._run_traces_begin(mode="eval")
        self._run_traces_on_epoch_begin(mode="eval", logs={"epoch": self.epoch})
        for eval_step, batch in enumerate(self.validation_fn().take(self.validation_steps)):
            self._run_traces_on_batch_begin(mode="eval", logs= {"epoch": self.epoch, "step": eval_step, "size": self.pipeline.batch_size})
            prediction, loss = self.forward_step(batch, mode="eval", epoch=self.epoch)
            self._run_traces_on_batch_end(mode="eval", logs= {"epoch": self.epoch, "step": eval_step, "size": self.pipeline.batch_size, "batch": batch, "prediction": prediction, "loss": loss})
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

    def _run_traces_end(self, mode):
        for trace in self.traces:
            trace.end(mode)

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

    # @tf.function
    def forward_step(self, batch, mode, epoch):
        prediction, losses = self.network.run_step(batch, mode, epoch)
        return prediction, losses