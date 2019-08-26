# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trace contains metrics and other information users want to track."""
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Trace:
    """Trace base class.
    User can use `Trace` to customize their own operations during training, validation and testing.
    The `Network` instance can be accessible by `self.network`.
    """
    def __init__(self, mode=None):
        """
        Args:
            mode: Restrict the trace to run only on given modes ('train', 'eval', 'test'). None will always execute
        """
        self.network = None
        self.mode = mode

    def on_begin(self):
        """Runs once at the beginning of training
        """
    def on_epoch_begin(self, state):
        """Runs at the beginning of each epoch of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "epoch_log": a dictionary that stores epoch related information, it can be used for sharing data between traces
        """
    def on_batch_begin(self, state):
        """Runs at the beginning of every batch of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "batch_idx": current local step of the epoch starting from 0
                * "batch_size": current global batch size
                * "batch": the batch data before the Network execution

        """
    def on_batch_end(self, state):
        """Runs at the end of every batch of the mode. Anything written to the top level of the state dictionary will be
            printed in the logs. Things written only to the batch sub-dictionary will not be logged

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "batch_idx": current local step of the epoch starting from 0
                * "batch_size": current global batch size
                * "batch": the batch data after the Network execution
        """
    def on_epoch_end(self, state):
        """Runs at the end of every epoch of the mode. Anything written into the state dictionary will be logged

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "epoch_log": a dictionary that stores epoch related information, it can be used for sharing data between traces
        """
    def on_end(self):
        """Runs once at the end training. Anything written into the state dictionary will be logged
        """


class Logger(Trace):
    """Logger, automatically applied by default.
    """
    def __init__(self, log_steps=100, train_watch_key=None, eval_average_key=None):
        """
        Args:
            log_steps (int, optional): Logging interval. Default value is 100.
        """
        super().__init__()
        self.log_steps = log_steps
        self.train_watch_key = self._initialize_keys(train_watch_key)
        self.eval_average_key = self._initialize_keys(eval_average_key)
        self.epochs_since_best = 0
        self.best_loss = None
        self.time_start = None

    def on_epoch_begin(self, state):
        self.eval_results = None

    def on_batch_begin(self, state):
        if state["mode"] == "train" and state["train_step"] % self.log_steps == 0:
            self.time_start = time.perf_counter()

    def on_batch_end(self, state):
        if state["mode"] == "train" and state["train_step"] % self.log_steps == 0:
            results = dict((key, state["batch"][key]) for key in self.train_watch_key if key in state["batch"])
            if state["train_step"] == 0:
                results["example_per_sec"] = 0.0
            else:
                results["example_per_sec"] = state["batch_size"] / (time.perf_counter() - self.time_start)
            self._print_message("FastEstimator-Train: step: {}; ".format(state["train_step"]), results)
        elif state["mode"] == "eval":
            if self.eval_results is None:
                self.eval_results = dict((key, [state["batch"][key]]) for key in self.eval_average_key if key in state["batch"])
            else:
                for key in self.eval_results.keys():
                    self.eval_results[key].append(state["batch"][key])

    def on_epoch_end(self, state):
        if state["mode"] == "eval":
            for key in self.eval_results.keys():
                self.eval_results[key] = np.mean(np.array(self.eval_results[key]), axis=0)
            #if there is only one loss, add several keys 
            state["epoch_log"].update(self.eval_results)
            self._print_message("FastEstimator-Eval: step: {}; ".format(state["train_step"]),  state["epoch_log"])

    @staticmethod
    def _initialize_keys(keys):
        if keys is None:
            keys = []
        elif not isinstance(keys, list):
            keys = [keys]
        return keys
    
    @staticmethod
    def _print_message(header, results):
        log_message = header
        for key in results.keys():
            if isinstance(results[key], np.ndarray):
                log_message += "\n{}:\n{};".format(key, np.array2string(results[key], separator=','))
            else:
                log_message += "{}: {}; ".format(key, str(results[key]))
        print(log_message)


class Accuracy(Trace):
    """Calculates accuracy for classification task and report it back to logger.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key, mode="eval", name="accuracy"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.total = 0
        self.correct = 0
        self.name = name

    def on_epoch_begin(self, state):
        self.total = 0
        self.correct = 0

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.correct += np.sum(prediction_label.ravel() == groundtruth_label.ravel())
        self.total += len(prediction_label.ravel())

    def on_epoch_end(self, state):
        state["epoch_info"][self.name] = self.correct / self.total


class ConfusionMatrix(Trace):
    """Computes confusion matrix between y_true and y_predict.

    Args:
        num_classes (int): Total number of classes of the confusion matrix.
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key, num_classes, mode="eval", output_name="confusion_matrix"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.confusion = None
        self.output_name = output_name

    def on_epoch_begin(self, state):
        self.confusion = None

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        batch_confusion = confusion_matrix(groundtruth_label, prediction_label, labels=list(range(0, self.num_classes)))
        if self.confusion is None:
            self.confusion = batch_confusion
        else:
            self.confusion += batch_confusion

    def on_epoch_end(self, state):
        state[self.output_name] = self.confusion


class Precision(Trace):
    """Calculates precision for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self,
                 true_key,
                 pred_key=None,
                 labels=None,
                 pos_label=1,
                 average='auto',
                 sample_weight=None,
                 mode="eval",
                 output_name="precision"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self.output_name = output_name

    def on_epoch_begin(self, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, state):
        if self.average == 'auto':
            if self.binary_classification:
                score = precision_score(np.ravel(self.y_true),
                                        np.ravel(self.y_pred),
                                        self.labels,
                                        self.pos_label,
                                        average='binary',
                                        sample_weight=self.sample_weight)
            else:
                score = precision_score(np.ravel(self.y_true),
                                        np.ravel(self.y_pred),
                                        self.labels,
                                        self.pos_label,
                                        average=None,
                                        sample_weight=self.sample_weight)
        else:
            score = precision_score(np.ravel(self.y_true),
                                    np.ravel(self.y_pred),
                                    self.labels,
                                    self.pos_label,
                                    self.average,
                                    self.sample_weight)
        state[self.output_name] = score


class Recall(Trace):
    """Calculates recall for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self,
                 true_key,
                 pred_key=None,
                 labels=None,
                 pos_label=1,
                 average='auto',
                 sample_weight=None,
                 mode="eval",
                 output_name="recall"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self.output_name = output_name

    def on_epoch_begin(self, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, state):
        if self.average == 'auto':
            if self.binary_classification:
                score = recall_score(np.ravel(self.y_true),
                                     np.ravel(self.y_pred),
                                     self.labels,
                                     self.pos_label,
                                     average='binary',
                                     sample_weight=self.sample_weight)
            else:
                score = recall_score(np.ravel(self.y_true),
                                     np.ravel(self.y_pred),
                                     self.labels,
                                     self.pos_label,
                                     average=None,
                                     sample_weight=self.sample_weight)
        else:
            score = recall_score(np.ravel(self.y_true),
                                 np.ravel(self.y_pred),
                                 self.labels,
                                 self.pos_label,
                                 self.average,
                                 self.sample_weight)
        state[self.output_name] = score


class F1Score(Trace):
    """Calculates F1 score for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self,
                 true_key,
                 pred_key=None,
                 labels=None,
                 pos_label=1,
                 average='auto',
                 sample_weight=None,
                 mode="eval",
                 output_name="f1score"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self.output_name = output_name

    def on_epoch_begin(self, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        assert prediction_label.size == groundtruth_label.size
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, state):
        if self.average == 'auto':
            if self.binary_classification:
                score = f1_score(np.ravel(self.y_true),
                                 np.ravel(self.y_pred),
                                 self.labels,
                                 self.pos_label,
                                 average='binary',
                                 sample_weight=self.sample_weight)
            else:
                score = f1_score(np.ravel(self.y_true),
                                 np.ravel(self.y_pred),
                                 self.labels,
                                 self.pos_label,
                                 average=None,
                                 sample_weight=self.sample_weight)
        else:
            score = f1_score(np.ravel(self.y_true),
                             np.ravel(self.y_pred),
                             self.labels,
                             self.pos_label,
                             self.average,
                             self.sample_weight)
        state[self.output_name] = score


class Dice(Trace):
    """Computes Dice score for binary classification between y_true and y_predict.

    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, threshold=0.5, mode="eval", output_name="dice"):
        super().__init__(mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.smooth = 1e-7
        self.threshold = threshold
        self.dice = None
        self.output_name = output_name

    def on_epoch_begin(self, state):
        self.dice = None

    def on_batch_end(self, state):
        groundtruth_label = np.array(state["batch"][self.true_key])
        if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(state["batch"][self.pred_key])
        prediction_label = (prediction_score >= self.threshold).astype(np.int)
        intersection = np.sum(groundtruth_label * prediction_label, axis=(1, 2, 3))
        area_sum = np.sum(groundtruth_label, axis=(1, 2, 3)) + np.sum(prediction_label, axis=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (area_sum + self.smooth)
        if self.dice is None:
            self.dice = dice
        else:
            self.dice = np.append(self.dice, dice, axis=0)

    def on_epoch_end(self, state):
        state[self.output_name] = np.mean(self.dice)
