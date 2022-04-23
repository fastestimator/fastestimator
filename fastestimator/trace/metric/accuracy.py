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
from typing import Iterable, Union

import numpy as np

from fastestimator.trace.meta._per_ds import per_ds
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@per_ds
@traceable()
class Accuracy(Trace):
    """A trace which computes the accuracy for a given set of predictions.

    Consider using MCC instead: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/

    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary.
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        from_logits: Whether y_pred is from logits. If True, a sigmoid will be applied to the prediction.
        output_name: What to call the output from this trace (for example in the logger output).
        per_ds: Whether to automatically compute this metric individually for every ds_id it runs on, in addition to
            computing an aggregate across all ds_ids on which it runs. This is automatically False if `output_name`
            contains a "|" character.
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[None, str, Iterable[str]] = ("eval", "test"),
                 ds_id: Union[None, str, Iterable[str]] = None,
                 from_logits: bool = False,
                 output_name: str = "accuracy",
                 per_ds: bool = True) -> None:
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name, ds_id=ds_id)
        self.from_logits = from_logits
        self.total = 0
        self.correct = 0
        self.per_ds = per_ds

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.total = 0
        self.correct = 0

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.shape[-1] > 1 and y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:  # binaray classification (pred shape is [batch, 1])
            if self.from_logits:
                y_pred = 1 / (1 + np.exp(-y_pred))
            y_pred = np.round(y_pred)
        assert y_pred.size == y_true.size
        self.correct += np.sum(y_pred.ravel() == y_true.ravel())
        self.total += len(y_pred.ravel())
        data.write_per_instance_log(self.outputs[0], np.array(y_pred.ravel() == y_true.ravel(), dtype=np.int8))

    def on_epoch_end(self, data: Data) -> None:
        data.write_with_log(self.outputs[0], self.correct / self.total)
