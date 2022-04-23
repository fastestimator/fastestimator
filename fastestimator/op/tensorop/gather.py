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
from typing import Any, Dict, Iterable, List, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend._gather_from_batch import gather_from_batch
from fastestimator.backend._reduce_max import reduce_max
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.base_util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class Gather(TensorOp):
    """Gather values from an input tensor.

    If indices are not provided, the maximum values along the batch dimension will be collected.

    Args:
        inputs: The tensor(s) to gather values from.
        indices: A tensor containing target indices to gather.
        outputs: The key(s) under which to save the output.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 indices: Union[None, str, List[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        indices = to_list(indices)
        self.num_indices = len(indices)
        combined_inputs = indices
        combined_inputs.extend(to_list(inputs))
        super().__init__(inputs=combined_inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        indices = data[:self.num_indices]
        inputs = data[self.num_indices:]
        results = []
        for idx, tensor in enumerate(inputs):
            # Check len(indices[0]) since an empty indices element is used to trigger the else
            if tf.is_tensor(indices[0]) or isinstance(indices[0], torch.Tensor):
                elem_len = indices[0].shape[0]
            else:
                elem_len = len(indices[0])
            if len(indices) > idx and elem_len > 0:
                results.append(gather_from_batch(tensor, indices=indices[idx]))
            elif len(indices) == 1 and elem_len > 0:
                # One set of indices for all outputs
                results.append(gather_from_batch(tensor, indices=indices[0]))
            else:
                results.append(reduce_max(tensor, 1))  # The maximum value within each batch element
        return results
