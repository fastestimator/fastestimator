import numpy as np

class Trace:
    # Trace is a combination of callback and metrics
    def __init__(self):
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
                "input": the pipeline output dictionary used as input of network
                "prediction": the batch predictions in dictionary format
                "loss": the batch loss (only available when mode is "train" or "eval")
        """
        pass

    def on_epoch_end(self, mode, logs):
        """this function runs at the end of every epoch, if needed to display metric in logger, then return the metric.
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
        """
        return None

    def end(self, mode):
        """this function runs at the end of the mode
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
        """
        pass

class Accuracy(Trace):
    def __init__(self, feature_true=None, feature_predict=None):
        self.feature_true = feature_true
        self.feature_predict = feature_predict

    def on_epoch_begin(self, mode, logs):
        self.total = 0
        self.correct = 0

    def on_batch_end(self, mode, logs):
        groundtruth_label = logs["input"][self.feature_true]
        if groundtruth_label.shape[-1] > 1:
            groundtruth_label = np.argmax(groundtruth_label, axis=-1)
        prediction_score = logs["prediction"]
        if isinstance(prediction_score, dict):
            prediction_score = prediction_score[self.feature_predict]
        binary_classification = prediction_score.shape[-1] == 1
        if binary_classification:
            prediction_label = np.round(prediction_score)
        else:
            prediction_label = np.argmax(prediction_score, axis=-1)
        assert prediction_label.size == groundtruth_label.size
        self.correct += np.sum(prediction_label.ravel() == groundtruth_label.ravel())
        self.total += len(prediction_label.ravel())

    def on_epoch_end(self, mode, logs):
        accuracy = self.correct/self.total
        return accuracy
