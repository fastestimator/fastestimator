from fastestimator.trace.trace import Trace
from fastestimator.backend.to_number import to_number
import numpy as np
import pdb

class Accuracy(Trace):
    def __init__(self, true_key, pred_key, log_names="accuracy"):
        super().__init__(inputs=(true_key, pred_key), mode="eval", log_names=log_names)
        self.total = 0
        self.correct = 0

    def on_epoch_begin(self):
        self.total = 0
        self.correct = 0
    
    def on_batch_end(self, data):
        y_true, y_pred = to_number(data[0]), to_number(data[1])
        if y_pred.shape[-1] == 1:
            label_pred = np.round(y_pred)
        else:
            label_pred = np.argmax(y_pred, axis=-1)
        assert label_pred.size == y_true.size
        self.correct += np.sum(label_pred.ravel() == y_true.ravel())
        self.total += len(label_pred.ravel())
    
    def on_epoch_end(self):
        self.system.add_buffer(self.log_names, self.correct / self.total)