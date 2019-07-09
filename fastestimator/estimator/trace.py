"""Trace contains metrics and other information users want to track."""
import time

import numpy as np
from sklearn.metrics import confusion_matrix


class Trace:
    """Trace base class.

    User can use `Trace` to customize their own operations during training, validation and testing.

    Args:
        network: `Network` instance can be accessible by `self.network`.
    """
    def __init__(self):
        self.network = None

    def begin(self, mode):
        """Runs at the beginning of the mode.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
        """

    def on_epoch_begin(self, mode, logs):
        """Runs at the beginning of each epoch of the mode.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
            logs (dict): Dictionary with the following key:

                * "epoch": current epoch index starting from 0
        """

    def on_batch_begin(self, mode, logs):
        """Runs at the beginning of every batch of the mode.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
            logs (dict): Dictionary with the following keys:

                * "epoch": current epoch index starting from 0
                * "step": current global step index starting from 0 (or batch index)
                * "size": current batch size
        """

    def on_batch_end(self, mode, logs):
        """Runs at the end of every batch of the mode.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
            logs (dict): Dictionary with the following keys:

                * "epoch": current epoch index starting from 0
                * "step": current global step index starting from 0 (or batch index)
                * "size": current batch size
                * "batch": the batch data used as input of network
                * "prediction": the batch predictions
                * "loss": the batch loss (only available when mode is "train" or "eval")
        """

    def on_epoch_end(self, mode, logs):
        """Runs at the end of every epoch of the mode.

        If needed to display metric in logger, then return the metric. The metric can be a scalar,
        list, tuple, numpy array or dictionary.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
            logs (dict): Dictionary with the following keys:

                * "epoch": current epoch index starting from 0
                * "loss": average loss within epoch (only available when mode is "eval")
        """

    def end(self, mode):
        """Runs at the end of the mode.

        Args:
            mode (str): Signature during different phases, can be "train", "eval" or "test".
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
        self.time_start = 0

    def on_epoch_begin(self, mode, logs):
        if mode == "train":
            self.time_start = time.time()

    def on_batch_end(self, mode, logs):
        if mode == "train" and logs["step"] % self.log_steps == 0:
            if logs["step"] == 0:
                example_per_sec = 0.0
            else:
                self.elapse_times.append(time.time() - self.time_start)
                example_per_sec = logs["size"] * self.log_steps / np.sum(self.elapse_times)
            loss = np.array(logs["loss"])
            print("FastEstimator-Train: step: %d; train_loss: %s; example/sec: %f;" %(logs["step"], str(loss), example_per_sec*self.num_process))
            self.elapse_times = []
            self.time_start = time.time()

    def on_epoch_end(self, mode, logs):
        if mode == "train":
            self.elapse_times.append(time.time() - self.time_start)
        elif mode == "eval":
            current_eval_loss = logs["loss"]
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
        y_true_key (str): Name of the keys in the ground truth label in data pipeline.
        y_pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. Default is `None`.
    """
    def __init__(self, y_true_key, y_pred_key=None):
        super().__init__()
        self.y_true_key = y_true_key
        self.y_pred_key = y_pred_key
        self.total = 0
        self.correct = 0

    def on_epoch_begin(self, mode, logs):
        if mode == "eval":
            self.total = 0
            self.correct = 0

    def on_batch_end(self, mode, logs):
        if mode == "eval":
            groundtruth_label = np.array(logs["batch"][self.y_true_key])
            if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
                groundtruth_label = np.argmax(groundtruth_label, axis=-1)
            prediction = logs["prediction"]
            if isinstance(prediction, dict):
                prediction_score = np.array(prediction[self.y_pred_key])
            else:

                prediction_score = np.array(prediction)
            binary_classification = prediction_score.shape[-1] == 1
            if binary_classification:
                prediction_label = np.round(prediction_score)
            else:
                prediction_label = np.argmax(prediction_score, axis=-1)
            assert prediction_label.size == groundtruth_label.size
            self.correct += np.sum(prediction_label.ravel() == groundtruth_label.ravel())
            self.total += len(prediction_label.ravel())

    def on_epoch_end(self, mode, logs):
        if mode == "eval":
            return self.correct/self.total
        return None

class ConfusionMatrix(Trace):
    """Computes confusion matrix between y_true and y_predict.

    Args:
        num_classes (int): Total number of classes of the confusion matrix.
        y_true_key (str): Name of the keys in the ground truth label in data pipeline.
        y_pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. Default is `None`.
    """
    def __init__(self, y_true_key, y_pred_key=None, num_classes=None):
        if not isinstance(num_classes, int):
            raise ValueError('num_classes should be a positive interger.')
        super().__init__()
        self.y_true_key = y_true_key
        self.y_pred_key = y_pred_key
        self.num_classes = num_classes
        self.confusion = None

    def on_epoch_begin(self, mode, logs):
        if mode == "eval":
            self.confusion = None

    def on_batch_end(self, mode, logs):
        if mode == "eval":
            groundtruth_label = np.array(logs["batch"][self.y_true_key])
            if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
                groundtruth_label = np.argmax(groundtruth_label, axis=-1)

            prediction = logs["prediction"]
            if isinstance(prediction, dict):
                prediction_score = np.array(prediction[self.y_pred_key])
            else:
                prediction_score = np.array(prediction)

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

    def on_epoch_end(self, mode, logs):
        if mode == "eval":
            return self.confusion
        return None

class Dice(Trace):
    """Computes Dice score for binary classification between y_true and y_predict.

    Args:
        y_true_key (str): Name of the keys in the ground truth label in data pipeline.
        y_pred_key (str, optional): If the network's output is a dictionary, name of the keys in predicted label. Default is `None`.
    """
    def __init__(self, y_true_key, y_pred_key=None, threshold=0.5):
        if not isinstance(int):
            raise ValueError('num_classes should be a positive interger.')
        super().__init__()
        self.y_true_key = y_true_key
        self.y_pred_key = y_pred_key
        self.smooth = 1e-7
        self.threshold = threshold
        self.dice = None

    def on_epoch_begin(self, mode, logs):
        if mode == "eval":
            self.dice = None

    def on_batch_end(self, mode, logs):
        if mode == "train":
            groundtruth_label = np.array(logs["batch"][self.y_true_key])
            if groundtruth_label.shape[-1] > 1 and groundtruth_label.ndim > 1:
                groundtruth_label = np.argmax(groundtruth_label, axis=-1)

            prediction = logs["prediction"]
            if isinstance(prediction, dict):
                prediction_score = np.array(prediction[self.y_pred_key])
            else:
                prediction_score = np.array(prediction)

            print('max score: {}'.format(prediction_score.max()))
            prediction_label = (prediction_score >= self.threshold).astype(np.int)

            intersection = np.sum(groundtruth_label * prediction_label, axis=(1, 2, 3))
            area_sum = np.sum(groundtruth_label, axis=(1, 2, 3)) + np.sum(prediction_label, axis=(1, 2, 3))
            dice = (2. * intersection + self.smooth) / (area_sum + self.smooth)
            if self.dice is None:
                self.dice = dice
            else:
                self.dice = np.append(self.dice, dice, axis=0)

    def on_epoch_end(self, mode, logs):
        if mode == "train":
            print(np.mean(self.dice))
            return np.mean(self.dice)
        return None
