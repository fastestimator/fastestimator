import numpy as np
import time

class Trace:
    def __init__(self):
        """trace is a combination of original callbacks, metrics and more, user can use trace to customize their own operations
        during training, validation and testing.

        network instance can be accessible by self.network

        """
        self.network = None
        pass

    def begin(self, mode):
        """this function runs at the beginning of the mode
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
        """
        pass

    def on_epoch_begin(self, mode, logs):
        """this function runs at the beginning of each epoch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
        """
        pass

    def on_batch_begin(self, mode, logs):
        """this function runs at the beginning of every batch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
                "step": current global step index starting from 0 (or batch index)
                "size": current batch size
        """
        pass

    def on_batch_end(self, mode, logs):
        """this function runs at the end of every batch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
                "step": current global step index starting from 0 (or batch index)
                "size": current batch size
                "batch": the batch data used as input of network
                "prediction": the batch predictions in dictionary format
                "loss": the batch loss (only available when mode is "train" or "eval")
        """
        pass

    def on_epoch_end(self, mode, logs):
        """this function runs at the end of every epoch, if needed to display metric in logger, then return the metric.
        the metric can be a scalar, list, tuple, numpy array or dictionary.
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
                "loss": average loss within epoch (only available when mode is "eval")
        """
        return None

    def end(self, mode):
        """this function runs at the end of the mode
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
        """
        pass


class TrainLogger(Trace):
    def __init__(self, log_steps=100, num_process=1):
        """Training logger, automatically applied by default
        
        Args:
            log_steps (int, optional): logging interval. Defaults to 100.
            num_process (int, optional): number of distributed training processes. Defaults to 1.
        """
        self.log_steps = log_steps
        self.num_process = num_process
        self.elapse_times = []
        self.epochs_since_best = 0
        self.best_loss = None

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
                if self.best_loss == None or current_eval_loss < self.best_loss:
                    self.best_loss = current_eval_loss
                    self.epochs_since_best = 0
                else:
                    self.epochs_since_best += 1
                output_metric["min_val_loss"] = self.best_loss
                output_metric["since_best"] = self.epochs_since_best
            return output_metric


class Accuracy(Trace):
    def __init__(self, feature_true=None, feature_predict=None):
        self.feature_true = feature_true
        self.feature_predict = feature_predict

    def on_epoch_begin(self, mode, logs):
        if mode == "eval":
            self.total = 0
            self.correct = 0

    def on_batch_end(self, mode, logs):
        if mode == "eval":
            groundtruth_label = np.array(logs["batch"][self.feature_true])
            if groundtruth_label.shape[-1] > 1 and len(groundtruth_label.shape) > 1:
                groundtruth_label = np.argmax(groundtruth_label, axis=-1)
            prediction = logs["prediction"]
            if isinstance(prediction, dict):
                prediction_score = np.array(prediction[self.feature_predict])
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
