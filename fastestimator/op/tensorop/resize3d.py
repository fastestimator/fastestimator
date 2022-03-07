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

import tensorflow as tf
import torch

from fastestimator.backend.resize3d import resize_3d
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class Resize3D(TensorOp):
    """
        Normalize a input tensor.

        Args:
            inputs: Key of the input tensor.
            outputs: Key of the output tensor.
            output_shape: output size of input tensor.
            mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
                regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
                like "!infer" or "!train".
            ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
                ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 output_shape: Sequence[float],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.output_shape = output_shape

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        return [resize_3d(elem, self.output_shape) for elem in data]
