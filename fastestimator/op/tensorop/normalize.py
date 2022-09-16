# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import Any, Dict, Iterable, List, Sequence, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._tensor_normalize import normalize
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class Normalize(TensorOp):
    """Normalize a input tensor (supports multi-io).

    Args:
        inputs: Key(s) of the input tensor that is to be normalized.
        outputs: Key(s) of the output tensor that has been normalized.
        mean: The mean which needs to applied (eg: None, 0.54, (0.24, 0.34, 0.35))
        std: The standard deviation which needs to applied (eg: None, 0.4, (0.1, 0.25, 0.45))
        max_pixel_value: The max value of the input data(eg: 255, 65025) to be multipled with mean and std to get actual mean and std.
                            To directly use the mean and std provide set max_pixel_value as 1.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mean: Union[float, Sequence[float]] = (0.485, 0.456, 0.406),
                 std: Union[float, Sequence[float]] = (0.229, 0.224, 0.225),
                 max_pixel_value: float = 255.0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.in_list, self.out_list = True, True

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        return [normalize(elem, self.mean, self.std, self.max_pixel_value) for elem in data]
