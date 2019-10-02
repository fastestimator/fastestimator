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


class Dice(Trace):
    """Computes Dice score for binary classification between y_true and y_predict.

    Args:
        true_key (str): Name of the keys in the ground truth label in data pipeline.
        pred_key (str, optional): Mame of the keys in predicted label. Default is `None`.
        threshold (float, optional): Threshold of the prediction. Defaults to 0.5.
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
                    execute. Defaults to 'eval'.
        output_name (str, optional): Name of the key to store to the state. Defaults to "dice".
    """
    def __init__(self, true_key, pred_key, threshold=0.5, mode="eval", output_name="dice"):
        super().__init__(outputs=output_name, mode=mode)
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
