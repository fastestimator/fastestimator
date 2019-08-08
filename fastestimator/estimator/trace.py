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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class Trace:
    """Trace base class.

    User can use `Trace` to customize their own operations during training, validation and testing.

    Args:
        network: `Network` instance can be accessible by `self.network`.
    """
    def __init__(self):
        self.network = None

    def begin(self, state):
        """Runs at the beginning of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode": the current run time mode, can be "train", "eval" or "test"
        """
    def on_epoch_begin(self, state):
        """Runs at the beginning of each epoch of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
        """
    def on_batch_begin(self, state):
        """Runs at the beginning of every batch of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "step": current global step index starting from 0 (or batch index)
                * "batch_size": current batch size on single machine
        """
    def on_batch_end(self, state):
        """Runs at the end of every batch of the mode.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "step": current global step index starting from 0 (or batch index)
                * "batch_size": current batch size on single machine
                * "batch": the batch data after the Network execution
                * "loss": the batch loss (only available when mode is "train" or "eval")
        """
    def on_epoch_end(self, state):
        """Runs at the end of every epoch of the mode.

        If needed to display metric in logger, then return the metric. The metric can be a scalar,
        list, tuple, numpy array or dictionary.

        Args:
            state (dict): dictionary of run time that has the following key(s):
                * "mode":  current run time mode, can be "train", "eval" or "test"
                * "epoch": current epoch index starting from 0
                * "loss": the batch loss (only available when mode is "train" or "eval")
        """
    def end(self, state):
        """Runs at the end of the mode.

        Args:
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
        self.elapse_times = []
        self.epochs_since_best = 0
        self.best_loss = None

    def on_epoch_begin(self, state):
        if state["mode"] == "train":
            self.time_start = time.time()

    def on_batch_end(self, state):
        if state["mode"] == "train" and state["step"] % self.log_steps == 0:
            if state["step"] == 0:
                example_per_sec = 0.0
            else:
                self.elapse_times.append(time.time() - self.time_start)
                example_per_sec = state["batch_size"] * self.log_steps / np.sum(self.elapse_times)
            loss = np.array(state["loss"])
            if loss.size == 1:
                loss = loss.ravel()[0]
            print("FastEstimator-Train: step: %d; train_loss: %s; example/sec: %.2f;" %
                  (state["step"], str(loss), example_per_sec * self.num_process))
            self.elapse_times = []
            self.time_start = time.time()

    def on_epoch_end(self, state):
        if state["mode"] == "train":
            self.elapse_times.append(time.time() - self.time_start)
        elif state["mode"] == "eval":
            current_eval_loss = state["loss"]
            if current_eval_loss.size == 1:
                current_eval_loss = current_eval_loss.ravel()[0]
            output_metric = {"val_loss": current_eval_loss}
            if np.isscalar(current_eval_loss):
                if self.best_loss is None or current_eval_loss < self.best_loss:
                    self.best_loss = current_eval_loss
                    self.epochs_since_best = 0
                else:
                    self.epochs_since_best += 1
                output_metric["min_val_loss"] = self.best_loss
                output_metric["since_best"] = self.epochs_since_best
            return output_metric
        return None


class Accuracy(Trace):
    """Calculates accuracy for classification task and report it back to logger.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key):
        self.true_key = true_key
        self.pred_key = pred_key
        self.total = 0
        self.correct = 0

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.total = 0
            self.correct = 0

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
        if state["mode"] == "eval":
            return self.correct / self.total
        else:
            return None


class ConfusionMatrix(Trace):
    """Computes confusion matrix between y_true and y_predict.

    Args:
        num_classes (int): Total number of classes of the confusion matrix.
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
    """
    def __init__(self, true_key, pred_key, num_classes):
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.confusion = None

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
            batch_confusion = confusion_matrix(groundtruth_label, prediction_label,
                                               labels=list(range(0, self.num_classes)))
            if self.confusion is None:
                self.confusion = batch_confusion
            else:
                self.confusion += batch_confusion

    def on_epoch_end(self, state):
        if state["mode"] == "eval":
            return self.confusion
        else:
            return None


class Precision(Trace):
    """Calculates precision for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label.
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, labels=None, pos_label=1, average='auto', sample_weight=None):
        super().__init__()
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.y_true = []
            self.y_pred = []

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
        if state["mode"] == "eval":
            if self.average == 'auto':
                if self.binary_classification:
                    return precision_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                           average='binary', sample_weight=self.sample_weight)
                else:
                    return precision_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                           average=None, sample_weight=self.sample_weight)
            else:
                return precision_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                       self.average, self.sample_weight)
        return None


class Recall(Trace):
    """Calculates recall for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. 
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, labels=None, pos_label=1, average='auto', sample_weight=None):
        super().__init__()
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.y_true = []
            self.y_pred = []

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
        if state["mode"] == "eval":
            if self.average == 'auto':
                if self.binary_classification:
                    return recall_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                        average='binary', sample_weight=self.sample_weight)
                else:
                    return recall_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                        average=None, sample_weight=self.sample_weight)
            else:
                return recall_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                    self.average, self.sample_weight)
        return None


class F1_score(Trace):
    """Calculates F1 score for classification task and report it back to logger.
    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. 
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, labels=None, pos_label=1, average='auto', sample_weight=None):
        super().__init__()
        self.true_key = true_key
        self.pred_key = pred_key
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.y_true = []
        self.y_pred = []

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.y_true = []
            self.y_pred = []

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
        if state["mode"] == "eval":
            if self.average == 'auto':
                if self.binary_classification:
                    return f1_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                    average='binary', sample_weight=self.sample_weight)
                else:
                    return f1_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label,
                                    average=None, sample_weight=self.sample_weight)
            else:
                return f1_score(np.ravel(self.y_true), np.ravel(self.y_pred), self.labels, self.pos_label, self.average,
                                self.sample_weight)
        return None


class Dice(Trace):
    """Computes Dice score for binary classification between y_true and y_predict.

    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. 
                                  Default is `None`.
    """
    def __init__(self, true_key, pred_key=None, threshold=0.5):
        super().__init__()
        self.true_key = true_key
        self.pred_key = pred_key
        self.smooth = 1e-7
        self.threshold = threshold
        self.dice = None

    def on_epoch_begin(self, state):
        if state["mode"] == "eval":
            self.dice = None

    def on_batch_end(self, state):
        if state["mode"] == "eval":
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
        if state["mode"] == "eval":
            return np.mean(self.dice)
