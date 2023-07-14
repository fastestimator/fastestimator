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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
import torch

from fastestimator.backend._expand_dims import expand_dims
from fastestimator.backend._get_shape import get_shape
from fastestimator.backend._squeeze import squeeze
from fastestimator.backend._to_tensor import to_tensor
from fastestimator.slicer.slicer import Slicer
from fastestimator.types import Tensor, TensorT
from fastestimator.util.base_util import to_list
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import get_num_gpus


@traceable()
class SlidingSlicer(Slicer):
    """A slicer which cuts inputs using a sliding window.

    This slicer cuts inputs using a sliding window, which should have the same rank as the input array. For example, if
    you want to feed a giant batched channel-first 2D image [B, C, W, H] into a model which can only handle smaller
    images of size [B, C, W2, H2] then you could set window_size=[-1, -1, W2, H2].

    Args:
        window_size: The size of your sliding window. Use -1 for any axis when you want to copy the size from the input.
        strides: How far to step in each direction each time you move a sliding window. Use 0 for each axis which you
            set as -1 in `window_size`. If you set `strides` < `window_size` you will get overlapping windows. If you
            set `strides` > `window_size` you will drop data from your input. If a single integer is provided if will be
            used for every axis where sliding occurs. If an empty sequence (default) is provided, then `strides` will be
            set equal to `window_size` such that there is no overlap between windows.
        pad_mode: One of 'drop', 'partial', or 'constant'. If 'drop' then the final slice along various axes will be
            dropped if the `window_size` does not tile perfectly with the input shape. If 'partial' then the final slice
            along various axes may be of a different (smaller) shape than other slices (if `window_size` does not
            perfectly tile the input shape). If 'constant' then the final slice along various axis will be filled out by
            `pad_val`.
        pad_val: The value to fill out the input with when `pad_mode` is 'constant'.
        slice: The input key(s) which this Slicer slices. Data which this slicer does not cut will be replicated across
            the resulting minibatches so that the network ops always have access to all of the batch keys. If multiple
            keys are provided here it is expected that they all have the same shape.
        unslice: The input key(s) which this Slicer un-slices. By default (empty tuple) the Slicer will un-slice
            whatever keys were specified in `slice`. If you do not want to un-slice those keys, then pass None or
            manually specify the specific key(s) which you would like this slicer to un-slice.
        unslice_mode: When re-combining your slices, how to you want to handle overlapping regions? Options are 'sum'
            or 'avg'. Regardless of option, any gaps between slices will be filled in by the `pad_val`.
        unslice_shape: The shape of your final tensor(s) after you unslice it. This argument is only required if you did
            not provide `slice` keys. If `slice` keys were provided, then the unslice_shape is inferred at every step
            based on the shape of the inputs before slicing.
        squeeze_window: If enabled, then all axes which are set to 1 in `window_size` will be squeezed out during the
            network forward pass, and re-introduced later during un-slicing. This makes it more convenient if you want
            to, for example, use SlidingSlicer to feed 3D image data into a 2D network (though you may want to instead
            consider AxisSlicer as a simpler way to achieve that purpose).
        mode: What mode(s) to invoke this Slicer in. For example, "train", "eval", "test", or "infer". To invoke
            regardless of mode, pass None. To invoke in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to invoke this Slicer in. To invoke regardless of ds_id, pass None. To invoke in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 window_size: Sequence[int],
                 strides: Union[int, Sequence[int]] = (),
                 pad_mode: str = 'drop',
                 pad_val: Union[int, float] = 0.0,
                 slice: Union[None, str, Sequence[str]] = None,
                 unslice: Union[None, str, Sequence[str]] = (),
                 unslice_mode: str = "avg",
                 unslice_shape: Optional[Tuple[int, ...]] = None,
                 squeeze_window: bool = False,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(slice=slice, unslice=unslice, mode=mode, ds_id=ds_id)
        self.window_size = to_list(window_size)
        assert len(self.window_size) > 0, "A window_size must be specified"
        for dim in self.window_size:
            assert isinstance(dim, int), \
                f"All window_size dimensions must be integers, but found {dim} of type {type(dim)}"
            assert dim > 0 or dim == -1, f"All window_size dimensions must be positive or -1, but found {dim}"
            # TODO - allow window_size=0 to delete an axis? or some other way to reduce dimensionality?
        if isinstance(strides, int):
            self.strides = [strides if size > 0 else 0 for size in self.window_size]
        else:
            self.strides = to_list(strides)
            if not strides:
                self.strides = [size if size > 0 else 0 for size in self.window_size]
        assert len(self.strides) == len(self.window_size), "strides and window_size should have the same number of " + \
            f"dimensions, but got {self.strides} vs {self.window_size}"
        for dim in self.strides:
            assert isinstance(dim, int), f"All stride dimensions must be integers, but found {dim} of type {type(dim)}"
            assert dim >= 0, f"All stride dimensions must be non-negative, but found {dim}."
        options = ('drop', 'partial', 'constant')
        assert pad_mode in options, f"pad_mode must be one of {options}, but got {pad_mode}"
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        options = ("sum", "avg")
        assert unslice_mode in options, f"unslice_mode must be one of: {options}, but got {unslice_mode}"
        self.unslice_mode = unslice_mode
        self.static_unslice_shape = True if unslice_shape is not None else False
        self.unslice_shape: Optional[List[int]] = to_list(unslice_shape)
        if self.unslice_inputs and not self.slice_inputs:
            assert self.static_unslice_shape, "If you want to use SlidingSlicer to unslice data you must either " + \
                "use it to perform the initial slicing, or else provide the unslice_shape that you want your data " + \
                "to take after it is reconstructed."
        self.auto_squeeze: List[int] = []
        if squeeze_window:
            for idx, size in enumerate(self.window_size):
                if size == 1:
                    self.auto_squeeze.append(idx)
        self.replica_batch_sizes: Dict[int, int] = {}

    def _slice_batch(self, batch: Tensor) -> List[Tensor]:
        shape = list(get_shape(batch))
        if len(self.window_size) != len(shape):
            raise ValueError(f"Sliding window size: {self.window_size} is incompatible with data shape: {shape}. " + \
                               "They must have the same rank.")
        if self.unslice_inputs and not self.static_unslice_shape:
            # If we have to unslice things later, then need to remember the desired shape if the user didn't give us one
            self.unslice_shape = [int(x) for x in tuple(shape)]
            if get_num_gpus() > 1:
                # Unfortunately in tf multi-gpu the batches are split over multiple replicas, in which case we need to
                # manually correct the desired batch dimension later
                replica_context = tf.distribute.get_replica_context()
                if replica_context is not None:
                    replica_id = int(replica_context.replica_id_in_sync_group)
                    self.replica_batch_sizes[replica_id] = int(shape[0])
        stride_template = [
            slice(None) if stride == 0 or stride >= dim else None for stride, dim in zip(self.strides, shape)
        ]
        cuts = self._get_cuts(shape, stride_template)
        batch = self._solve_padding(batch=batch, batch_shape=shape)
        minibatches = [batch[cut] for cut in cuts]
        for ax in reversed(self.auto_squeeze):
            minibatches = [squeeze(minibatch, axis=ax) for minibatch in minibatches]
        return minibatches

    def _get_cuts(self, data_shape: List[int], stride_template: List[Optional[slice]]) -> List[List[slice]]:
        results = []
        for axis, cut_template in enumerate(stride_template):
            if cut_template is None:
                target_shape = data_shape[axis]
                for start in range(0, target_shape, self.strides[axis]):
                    stop = start + self.window_size[axis]
                    if stop > target_shape:
                        if self.pad_mode == 'drop':
                            continue
                        elif self.pad_mode == 'partial':
                            stop = target_shape
                        else:
                            # Padding the input
                            pass
                    new_cut = slice(start, stop)
                    new_template = list(stride_template)
                    new_template[axis] = new_cut
                    results.extend(self._get_cuts(data_shape, new_template))
                # Each level of recursion should only solve one axis, so break here
                break
        if not results:
            # No slices were generated, so all of the slices are already defined within the stride_template
            return [stride_template]
        return results

    def _solve_padding(self, batch: TensorT, batch_shape: List[int]) -> TensorT:
        if self.pad_mode == 'constant':
            paddings = [[0, int(win - ((tru % (stride or tru)) or win))] for tru,
                        win,
                        stride in zip(batch_shape, self.window_size, self.strides)]
            if isinstance(batch, torch.Tensor):
                paddings.reverse()  # Torch padding reads from right-most dim to left-most dim
                paddings = [elem for x in paddings for elem in x]
                batch = torch.nn.functional.pad(batch, pad=paddings, mode='constant', value=self.pad_val)
            else:
                batch = tf.pad(batch,
                               paddings=tf.constant(paddings),
                               mode="CONSTANT",
                               constant_values=tf.cast(self.pad_val, dtype=batch.dtype))
        return batch

    def _unslice_batch(self, slices: Tuple[Tensor, ...], key: str) -> Tensor:
        target_shape = self.unslice_shape
        assert target_shape is not None, "Unit tests should run forward_slicers before running reverse_slicers"
        if self.replica_batch_sizes:
            target_shape[0] = sum(self.replica_batch_sizes.values())

        stride_template = [
            slice(None) if stride == 0 or stride >= dim else None for stride, dim in zip(self.strides, target_shape)
        ]
        cuts = self._get_cuts(target_shape, stride_template)
        assert len(cuts) == len(slices), f"SlidingSlicer could not unslice key: {key}. It received {len(slices)} " + \
            f"slices, but a target_shape of {target_shape} with strides of {self.strides} could not have produced this."

        # TF doesn't support slice assignment, and np doesn't support indexing using list of slice, so we'll use torch
        # for everything and then cast back to tf later if needed
        merged_dtype = torch.float32 if self.unslice_mode == 'avg' else to_tensor(slices[0], target_type='torch').dtype
        merged = torch.zeros(size=target_shape, dtype=merged_dtype)
        merged = self._solve_padding(batch=merged, batch_shape=target_shape)
        counts = torch.zeros(size=target_shape, dtype=torch.int32)
        counts = self._solve_padding(batch=counts, batch_shape=target_shape)

        for cut, data in zip(cuts, slices):
            for ax in self.auto_squeeze:
                data = expand_dims(data, axis=ax)
            merged[cut] += to_tensor(data=data, target_type='torch')
            counts[cut] += 1

        if self.unslice_mode == 'avg':
            merged /= torch.maximum(counts, torch.tensor(1.0))

        merged[counts == 0] = torch.tensor(self.pad_val, dtype=merged_dtype)

        # Remove any border padding which may have been added
        merged = merged[[slice(target) for target in target_shape]]

        return merged if isinstance(slices[0], torch.Tensor) else to_tensor(data=merged, target_type='tf')
