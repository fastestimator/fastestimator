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
import math
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from fastestimator.util.op import Op


class Trace:
    """Trace base class.
    User can use `Trace` to customize their own operations during training, validation and testing.
    The `Network` instance can be accessible by `self.network`.
    """
    def __init__(self):
        self.network = None
        self.begin = None
        self.epoch_begin = None
        self.batch_begin = None
        self.batch_end = None
        self.epoch_end = None
        self.end = None

    def _init_begin(self, inputs=None, outputs=None, mode=None):
        self.begin = Op(inputs=inputs, outputs=outputs, mode=mode)

    def _init_epoch_begin(self, inputs=None, outputs=None, mode=None):
        self.epoch_begin = Op(inputs=inputs, outputs=outputs, mode=mode)

    def _init_batch_begin(self, inputs=None, outputs=None, mode=None):
        self.batch_begin = Op(inputs=inputs, outputs=outputs, mode=mode)

    def _init_batch_end(self, inputs=None, outputs=None, mode=None):
        self.batch_end = Op(inputs=inputs, outputs=outputs, mode=mode)

    def _init_epoch_end(self, inputs=None, outputs=None, mode=None):
        self.epoch_end = Op(inputs=inputs, outputs=outputs, mode=mode)

    def _init_end(self, inputs=None, outputs=None, mode=None):
        self.end = Op(inputs=inputs, outputs=outputs, mode=mode)

    def on_begin(self, data, state):
        """
        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode": the current run time mode, can be "train", "eval" or "test"
        """
    def on_epoch_begin(self, data, state):
        """Runs at the beginning of each epoch of the mode.

        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
        """
    def on_batch_begin(self, data, state):
        """Runs at the beginning of every batch of the mode.

        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "batch_idx": current local (within-batch) evaluation step starting from 0
                * "batch_size": current batch size per gpu

        """
    def on_batch_end(self, data, state):
        """Runs at the end of every batch of the mode.

        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "batch_idx": current local (within-batch) evaluation step starting from 0
                * "batch_size": current batch size on single machine
                * "loss": the batch loss (only available when mode is "train" or "eval")
        """
    def on_epoch_end(self, data, state):
        """Runs at the end of every epoch of the mode.

        If needed to display metric in logger, then return the metric. The metric can be a scalar,
        list, tuple, numpy array or dictionary.

        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "train_step": current global training step starting from 0
                * "loss": the average loss of all batches (only available when mode is "train" or "eval")
        """
    def on_end(self, data, state):
        """Runs at the end of the mode.

        Args:
            data: the elements from the execution dictionary corresponding to this Traces' input keys
            state (dict): dictionary of run time that has the following key(s):
                * "mode": the current run time mode, can be "train", "eval" or "test"
        """


class TrainLogger(Trace):
    """Training logger, automatically applied by default.
    """
    def __init__(self, log_steps=100, num_process=1):
        """
        Args:
            log_steps (int, optional): Logging interval. Default value is 100.
            num_process (int, optional): Number of distributed training processes. Default is 1.
        """
        super().__init__()
        self.log_steps = log_steps
        self.num_process = num_process
        self.epochs_since_best = 0
        self.best_loss = None
        self.time_start = None
        self.best_loss = math.inf
        self.epochs_since_best = 0
        self._init_batch_begin(mode="train")
        self._init_batch_end(outputs=("train_loss", "examples/sec"), mode="train")
        self._init_epoch_end(outputs=("val_loss", "min_val_loss", "since_best"), mode="eval")

    def on_batch_begin(self, data, state):
        if state["train_step"] % self.log_steps == 0:
            self.time_start = time.perf_counter()

    def on_batch_end(self, data, state):
        if state["train_step"] % self.log_steps == 0:
            if state["train_step"] == 0:
                example_per_sec = 0.0
            else:
                example_per_sec = state["batch_size"] / (time.perf_counter() - self.time_start)
            loss = np.array(state["loss"])
            if loss.size == 1:
                loss = loss.ravel()[0]
            return loss, round(example_per_sec * self.num_process, 2)

    def on_epoch_end(self, data, state):
        current_eval_loss = state["loss"]
        if current_eval_loss.size == 1:
            current_eval_loss = current_eval_loss.ravel()[0]
        if np.isscalar(current_eval_loss):
            if current_eval_loss < self.best_loss:
                self.best_loss = current_eval_loss
                self.epochs_since_best = 0
            else:
                self.epochs_since_best += 1
        return current_eval_loss, self.best_loss, self.epochs_since_best


class Accuracy(Trace):
    """Calculates accuracy for classification task and report it back to logger.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key, outputs="accuracy", mode="eval"):
        super().__init__()
        self.total = 0
        self.correct = 0
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(mode=mode, outputs=outputs)

    def on_epoch_begin(self, data, state):
        self.total = 0
        self.correct = 0

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.correct += np.sum(prediction_label.ravel() == groundtruth_label.ravel())
        self.total += len(prediction_label.ravel())

    def on_epoch_end(self, data, state):
        return self.correct / self.total


class ConfusionMatrix(Trace):
    """Computes confusion matrix between y_true and y_predict.

    Args:
        num_classes (int): Total number of classes of the confusion matrix.
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key, num_classes, outputs="confusion_matrix", mode="eval"):
        super().__init__()
        self.num_classes = num_classes
        self.confusion = None
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(outputs=outputs, mode=mode)

    def on_epoch_begin(self, data, state):
        self.confusion = None

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
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

    def on_epoch_end(self, data, state):
        return self.confusion


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
                 outputs='precision',
                 mode='eval'):
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(outputs=outputs, mode=mode)

    def on_epoch_begin(self, data, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, data, state):
        if self.average == 'auto':
            if self.binary_classification:
                return precision_score(np.ravel(self.y_true),
                                       np.ravel(self.y_pred),
                                       self.labels,
                                       self.pos_label,
                                       average='binary',
                                       sample_weight=self.sample_weight)
            else:
                return precision_score(np.ravel(self.y_true),
                                       np.ravel(self.y_pred),
                                       self.labels,
                                       self.pos_label,
                                       average=None,
                                       sample_weight=self.sample_weight)
        else:
            return precision_score(np.ravel(self.y_true),
                                   np.ravel(self.y_pred),
                                   self.labels,
                                   self.pos_label,
                                   self.average,
                                   self.sample_weight)


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
                 outputs="recall",
                 mode="eval"):
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(outputs=outputs, mode=mode)

    def on_epoch_begin(self, data, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, data, state):
        if self.average == 'auto':
            if self.binary_classification:
                return recall_score(np.ravel(self.y_true),
                                    np.ravel(self.y_pred),
                                    self.labels,
                                    self.pos_label,
                                    average='binary',
                                    sample_weight=self.sample_weight)
            else:
                return recall_score(np.ravel(self.y_true),
                                    np.ravel(self.y_pred),
                                    self.labels,
                                    self.pos_label,
                                    average=None,
                                    sample_weight=self.sample_weight)
        else:
            return recall_score(np.ravel(self.y_true),
                                np.ravel(self.y_pred),
                                self.labels,
                                self.pos_label,
                                self.average,
                                self.sample_weight)


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
                 outputs="f1score",
                 mode='eval'):
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []
        self.binary_classification = None
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(outputs=outputs, mode=mode)

    def on_epoch_begin(self, data, state):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        self.binary_classification = binary_classification or prediction_score.shape[-1] == 2
        assert prediction_label.size == groundtruth_label.size
        self.y_pred.append(list(prediction_label.ravel()))
        self.y_true.append(list(groundtruth_label.ravel()))

    def on_epoch_end(self, data, state):
        if self.average == 'auto':
            if self.binary_classification:
                return f1_score(np.ravel(self.y_true),
                                np.ravel(self.y_pred),
                                self.labels,
                                self.pos_label,
                                average='binary',
                                sample_weight=self.sample_weight)
            else:
                return f1_score(np.ravel(self.y_true),
                                np.ravel(self.y_pred),
                                self.labels,
                                self.pos_label,
                                average=None,
                                sample_weight=self.sample_weight)
        else:
            return f1_score(np.ravel(self.y_true),
                            np.ravel(self.y_pred),
                            self.labels,
                            self.pos_label,
                            self.average,
                            self.sample_weight)


class Dice(Trace):
    """Computes Dice score for binary classification between y_true and y_predict.

    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, threshold=0.5, outputs="dice", mode='eval'):
        super().__init__()
        self.smooth = 1e-7
        self.threshold = threshold
        self.dice = None
        self._init_epoch_begin(mode=mode)
        self._init_batch_end(inputs=(true_key, pred_key), mode=mode)
        self._init_epoch_end(outputs=outputs, mode=mode)

    def on_epoch_begin(self, data, state):
        self.dice = None

    def on_batch_end(self, data, state):
        true, pred = data
        groundtruth_label = np.array(true)
        if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = np.array(pred)
        prediction_label = (prediction_score >= self.threshold).astype(np.int)
        intersection = np.sum(groundtruth_label * prediction_label, axis=(1, 2, 3))
        area_sum = np.sum(groundtruth_label, axis=(1, 2, 3)) + np.sum(prediction_label, axis=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (area_sum + self.smooth)
        if self.dice is None:
            self.dice = dice
        else:
            self.dice = np.append(self.dice, dice, axis=0)

    def on_epoch_end(self, data, state):
        return np.mean(self.dice)
