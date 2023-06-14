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
from typing import Iterable, List, Sequence, Tuple, Union

from fastestimator.backend._concat import concat
from fastestimator.backend._expand_dims import expand_dims
from fastestimator.backend._get_shape import get_shape
from fastestimator.slicer.slicer import Slicer
from fastestimator.types import Tensor
from fastestimator.util.traceability_util import traceable


@traceable()
class AxisSlicer(Slicer):
    """A slicer which cuts along a given axis.

    This slicer cuts volumes along the specified axis, reducing the total dimension of the input by 1. For example, if
    you want to feed a batched channel-first 3D volume [B, C, W, H, D] into a 2D model [B, C, W, H] then you could set
    `axis=-1` or `axis=4`.

    Args:
        slice: The input key(s) which this Slicer slices. Data which this slicer does not cut will be replicated across
            the resulting minibatches so that the network ops always have access to all of the batch keys.
        unslice: The input key(s) which this Slicer un-slices. By default (empty tuple) the Slicer will un-slice
            whatever keys were specified in `slice`. If you do not want to un-slice those keys, then pass None or
            manually specify the specific key(s) which you would like this slicer to un-slice.
        axis: The axis along which to cut the data
        mode: What mode(s) to invoke this Slicer in. For example, "train", "eval", "test", or "infer". To invoke
            regardless of mode, pass None. To invoke in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to invoke this Slicer in. To invoke regardless of ds_id, pass None. To invoke in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 axis: int,
                 slice: Union[None, str, Sequence[str]] = None,
                 unslice: Union[None, str, Sequence[str]] = (),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(slice=slice, unslice=unslice, mode=mode, ds_id=ds_id)
        assert isinstance(axis, int), f"Axis must be an integer, got {type(axis)}"
        self.axis = axis

    def _slice_batch(self, batch: Tensor) -> List[Tensor]:
        shape = get_shape(batch)
        cut_index: List[Union[slice, int]] = [slice(None) for _ in range(len(shape))]
        slices = []
        for i in range(0, shape[self.axis]):
            cut_index[self.axis] = i
            slices.append(batch[cut_index])
        return slices

    def _unslice_batch(self, slices: Tuple[Tensor, ...], key: str) -> Tensor:
        expanded = [expand_dims(elem, self.axis) for elem in slices]
        return concat(expanded, axis=self.axis)
