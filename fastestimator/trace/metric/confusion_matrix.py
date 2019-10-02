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
import numpy as np
from sklearn.metrics import confusion_matrix

from fastestimator.trace import Trace


class ConfusionMatrix(Trace):
    """Computes confusion matrix between y_true and y_predict.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
        num_classes (int): Total number of classes of the confusion matrix.
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
        output_name (str, optional): Name of the key to store to the state. Defaults to "confusion_matrix".
    """
    def __init__(self, true_key, pred_key, num_classes, mode="eval", output_name="confusion_matrix"):
        super().__init__(outputs=output_name, mode=mode)
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
