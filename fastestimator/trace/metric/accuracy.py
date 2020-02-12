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

from typing import List, TypeVar

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.to_number import to_number
from fastestimator.trace.trace import Trace

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class Accuracy(Trace):
    def __init__(self, true_key, pred_key, output_name="accuracy"):
        super().__init__(inputs=(true_key, pred_key), mode="eval", outputs=output_name)
        self.total = 0
        self.correct = 0

    def on_epoch_begin(self):
        self.total = 0
        self.correct = 0

    def on_batch_end(self, data: List[Tensor]):
        y_true, y_pred = to_number(data[0]), to_number(data[1])
        if y_pred.shape[-1] == 1:
            label_pred = np.round(y_pred)
        else:
            label_pred = np.argmax(y_pred, axis=-1)
        assert label_pred.size == y_true.size
        self.correct += np.sum(label_pred.ravel() == y_true.ravel())
        self.total += len(label_pred.ravel())

    def on_epoch_end(self):
        self.system.add_buffer(self.outputs, self.correct / self.total)
