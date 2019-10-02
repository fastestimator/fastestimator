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

from fastestimator.trace import Trace


class Accuracy(Trace):
    """Calculates accuracy for classification task and report it back to logger.

    Args:
        true_key (str): Name of the key that corresponds to ground truth in batch dictionary
        pred_key (str): Name of the key that corresponds to predicted score in batch dictionary
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
        output_name (str, optional): Name of the key to store to the state. Defaults to "accuracy".
    """
    def __init__(self, true_key, pred_key, mode="eval", output_name="accuracy"):

        super().__init__(outputs=output_name, mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.total = 0
        self.correct = 0
        self.output_name = output_name

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
        state[self.output_name] = self.correct / self.total
