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
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_std import reduce_std
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class Normalize(TensorOp):
    """Normalize a input tensor.

    Args:
        inputs: Key of the input tensor that is to be normalized.
        outputs: Key of the output tensor that has been normalized.
        eps: Min value to be added to avoid divide by zero error.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 eps: float = 1e-6,
                 mean: Union[None, float, Tuple[float, ...]] = None, 
                 std: Union[None, float, Tuple[float, ...]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        if mean is None:
            self.mean = reduce_mean(inputs, axis=[0, 1, 2])
        else:
            self.mean = mean
        
        if std is None:
            self.std = reduce_std(inputs, axis=[0, 1, 2])
        else:
            self.std = std
        self.eps = eps 

    def forward(self, data: List[Tensor]) -> List[Tensor]:
        return [reshape(item, self.eps, self.mean, self.std) for item in data]
