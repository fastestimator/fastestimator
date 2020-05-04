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
from typing import List, Union

import numpy as np

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace
from fastestimator.util import Data


class Dice(Trace):
    """Dice score for binary classification between y_true and y_predicted.

    Args:
        true_key: The key of the ground truth mask.
        pred_key: The key of the prediction values.
        threshold: The threshold for binarizing the prediction.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        output_name: What to call the output from this trace (for example in the logger output).
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 threshold: float = 0.5,
                 mode: Union[None, str, List[str]] = ("eval", "test"),
                 output_name: str = "Dice") -> None:
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name)
        self.threshold = threshold
        self.smooth = 1e-8
        self.dice = np.array([])

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.dice = []

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        batch_size = y_true.shape[0]
        y_true, y_pred = y_true.reshape((batch_size, -1)), y_pred.reshape((batch_size, -1))

        prediction_label = (y_pred >= self.threshold).astype(np.int32)

        intersection = np.sum(y_true * prediction_label, axis=-1)
        area_sum = np.sum(y_true, axis=-1) + np.sum(prediction_label, axis=-1)
        dice = (2. * intersection + self.smooth) / (area_sum + self.smooth)
        self.dice.extend(list(dice))

    def on_epoch_end(self, data: Data) -> None:
        data.write_with_log(self.outputs[0], np.mean(self.dice))
