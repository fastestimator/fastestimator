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
from typing import Union, Iterable, TypeVar, List, Dict, Any

import tensorflow as tf
import torch

from fastestimator.backend.watch import watch
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class Watch(TensorOp):
    """Watch one or more tensors for later gradient computation.

    Args:
        inputs: which tensors to watch during future computation.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self, inputs: Union[None, str, Iterable[str]], mode: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=inputs, mode=mode)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        for idx, tensor in enumerate(data):
            data[idx] = watch(tensor=tensor, tape=state['tape'])
        return data
