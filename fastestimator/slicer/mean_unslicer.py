# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
from typing import Iterable, Sequence, Tuple, Union

from fastestimator.backend._zeros_like import zeros_like
from fastestimator.slicer.slicer import Slicer
from fastestimator.types import Tensor
from fastestimator.util.traceability_util import traceable


@traceable()
class MeanUnslicer(Slicer):
    """A slicer which re-combines mini-batches via averaging.

    Args:
        unslice: The input key(s) which this Slicer un-slices.
        axis: The axis along which to cut the data
        mode: What mode(s) to invoke this Slicer in. For example, "train", "eval", "test", or "infer". To invoke
            regardless of mode, pass None. To invoke in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to invoke this Slicer in. To invoke regardless of ds_id, pass None. To invoke in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 unslice: Union[str, Sequence[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(slice=None, unslice=unslice, mode=mode, ds_id=ds_id)

    def _unslice_batch(self, slices: Tuple[Tensor, ...], key: str) -> Tensor:
        mean = zeros_like(slices[0])
        for minibatch in slices:
            mean += minibatch
        mean /= len(slices)
        return mean
