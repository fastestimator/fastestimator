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
from typing import Union, Iterable

import numpy as np

from fastestimator.trace.meta.per_ds import per_ds
from fastestimator.trace.trace import Trace
from fastestimator.util import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@per_ds
@traceable()
class Dice(Trace):
    """Dice score for binary classification between y_true and y_predicted.

    Args:
        true_key: The key of the ground truth mask.
        pred_key: The key of the prediction values.
        threshold: The threshold for binarizing the prediction.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        output_name: What to call the output from this trace (for example in the logger output).
        per_ds: Whether to automatically compute this metric individually for every ds_id it runs on, in addition to
            computing an aggregate across all ds_ids on which it runs. This is automatically False if `output_name`
            contains a "|" character.
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 threshold: float = 0.5,
                 mode: Union[None, str, Iterable[str]] = ("eval", "test"),
                 ds_id: Union[None, str, Iterable[str]] = None,
                 output_name: str = "Dice",
                 per_ds: bool = True) -> None:
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name, ds_id=ds_id)
        self.threshold = threshold
        self.smooth = 1e-8
        self.dice = []
        self.per_ds = per_ds

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
