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
from typing import Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple, Union

import tensorflow as tf
from tensorflow.python.distribute.values import DistributedValues

from fastestimator.types import Tensor
from fastestimator.util.base_util import check_ds_id, check_io_names, parse_modes, to_list, to_set
from fastestimator.util.traceability_util import traceable


@traceable()
class Slicer():
    """A base class for FastEstimator Slicers.

    Slicers cut batches into mini-batches in order to pass them through the network, then re-assemble them after
    bringing all of the pieces back together on the CPU before handing them off to network post-processing and traces.

    Args:
        slice: The input key(s) which this Slicer slices. Data which this slicer does not cut will be replicated across
            the resulting minibatches so that the network ops always have access to all of the batch keys.
        unslice: The input key(s) which this Slicer un-slices. By default (empty tuple) the Slicer will un-slice
            whatever keys were specified in `slice`. If you do not want to un-slice those keys, then pass None or
            manually specify the specific keys which you would like this slicer to un-slice.
        mode: What mode(s) to invoke this Slicer in. For example, "train", "eval", "test", or "infer". To invoke
            regardless of mode, pass None. To invoke in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to invoke this Slicer in. To invoke regardless of ds_id, pass None. To invoke in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    slice_inputs: List[str]
    unslice_inputs: List[str]
    mode: Set[str]
    ds_id: Set[str]
    minibatch_size: int = 0  # Used for traceability

    def __init__(self,
                 slice: Union[None, str, Sequence[str]] = None,
                 unslice: Union[None, str, Sequence[str]] = (),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        if isinstance(unslice, tuple) and len(unslice) == 0:
            # If unslice keys are not specified, then use the slice keys by default for convenience
            unslice = slice
        self.slice_inputs = check_io_names(to_list(slice))
        self.unslice_inputs = check_io_names(to_list(unslice))
        self.mode = parse_modes(to_set(mode))
        self.ds_id = check_ds_id(to_set(ds_id))
        self.minibatch_size = 0
        if not self.slice_inputs and not self.unslice_inputs:
            raise ValueError("At least one of slice_inputs or unslice_inputs should be provided")
        if self.slice_inputs and type(self)._slice_batch == Slicer._slice_batch:
            raise NotImplementedError(
                f"Slice inputs were provided, but {type(self).__name__} does not implement _slice_batch")
        if self.unslice_inputs and type(self)._unslice_batch == Slicer._unslice_batch:
            raise NotImplementedError(
                f"Unslice inputs were provided, but {type(self).__name__} does not implement _unslice_batch")

    def slice_batches(self, batches: Tuple[Tensor, ...]) -> List[Tuple[Tensor, ...]]:
        """A method to convert one or more data tensors into slices.

        Args:
            batches: One or more data tensors, in a 1-1 relationship with self.slice_inputs

        Returns:
            The slices corresponding to each of the input batch(es)
        """
        slices = [self._slice_batch(batch) for batch in batches]
        self.minibatch_size = len(slices[0])
        for sl in slices[1:]:
            assert len(sl) == self.minibatch_size, \
                f"Slicer produced inconsistent number of slices over inputs: {self.slice_inputs}"
        return [minibatch for minibatch in zip(*slices)]

    def _slice_batch(self, batch: Tensor) -> List[Tensor]:
        """A method to convert a single tensor into a list of smaller slices.

        Child classes which support slice_inputs must implement this method.

        Args:
            batch: A tensor of data to be cut apart.

        Returns:
            The input tensor, but cut into slices
        """
        raise NotImplementedError

    def _unslice_batch(self, slices: Tuple[Tensor, ...], key: str) -> Tensor:
        """A method to convert a list of smaller slices back into a single tensor.

        Child classes which support unslice_inputs must implement this method.

        Args:
            slices: Small slices of data to be re-combined
            key: The data dictionary key corresponding to these slices (in case you need to use it for something). Note
                that slicers are not guaranteed to be provided with all of their declared unslice_inputs during each
                step. If the key is not used later on during training, then we do not bother un-slicing it.

        Returns:
            A single tensor produced by combining all of the available slices.
        """
        raise NotImplementedError


def sanity_assert_slicers(slicers: List[Slicer]) -> None:
    """A sanity test to ensure that slicers in a given list don't interfere with each-other.

    Args:
        slicers: The slicers to be run during a given batch.

    Raises:
        ValueError: If multiple slicers attempt to slice/unslice the same keys simultaneously.
    """
    if len(slicers) == 0:
        return
    slice_inputs = set(slicers[0].slice_inputs)
    unslice_inputs = set(slicers[0].unslice_inputs)
    for slicer in slicers[1:]:
        more_s_inputs = set(slicer.slice_inputs)
        more_u_inputs = set(slicer.unslice_inputs)
        if slice_inputs & more_s_inputs:
            raise ValueError(
                f"Multiple Slicers tried to slice the same keys simultaneously: {slice_inputs & more_s_inputs}")
        if unslice_inputs & more_u_inputs:
            raise ValueError(
                f"Multiple Slicers tried to un-slice the same keys simultaneously: {unslice_inputs & more_u_inputs}")
        slice_inputs |= more_s_inputs
        unslice_inputs |= more_u_inputs
    if unslice_inputs and not slice_inputs:
        raise ValueError("Cannot unslice keys if no slicing is performed.")


def forward_slicers(slicers: List[Slicer], data: MutableMapping[str, Tensor]) -> List[Dict[str, Tensor]]:
    """Perform a forward pass over a list of slicers, cutting a batch of data apart into multiple mini-batches.

    Any data (keys) which are not explicitly handled by a Slicer in the input list will simply be replicated across
    all of the mini-batches produced by this function.

    Args:
        slicers: The slicers with which to cut apart the batch.
        data: The batch to be cut apart.

    Raises:
        ValueError: If the slicers produce an inconsistent number of mini-batches.

    Returns:
        A list of mini-batches created by slicing the input batch.
    """
    slices = []
    sliced_inputs = set()
    for slicer in slicers:
        if not slicer.slice_inputs:
            continue
        input_data = tuple([data[key] for key in slicer.slice_inputs])
        if input_data and isinstance(input_data[0], DistributedValues):
            strategy = tf.distribute.get_strategy()
            sliced_data = strategy.extended.call_for_each_replica(fn=slicer.slice_batches, args=(input_data, ))
        else:
            sliced_data = slicer.slice_batches(input_data)
        if not slices:
            slices = [{key: value for key, value in zip(slicer.slice_inputs, element)} for element in sliced_data]
        else:
            if len(sliced_data) != len(slices):
                raise ValueError(
                    f"Multiple Slicers produced an inconsistent number of slices: {len(slices)} vs {len(sliced_data)}")
            for minibatch, new_entry in zip(slices, [{key: value for key, value in zip(slicer.slice_inputs, element)}
                                                     for element in sliced_data]):
                minibatch.update(new_entry)
        sliced_inputs |= set(slicer.slice_inputs)
    leftover_data = {key: data[key] for key in data.keys() - sliced_inputs}
    for minibatch in slices:
        minibatch.update(leftover_data)
    return slices


def reverse_slicers(slicers: List[Slicer], data: List[MutableMapping[str, Tensor]],
                    original_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Compile a list of mini-batches back into a single batch of data, according to a given list of slicers.

    Note that the `data` provided need not contain all of the keys requested by the slicer.unslice_inputs. Missing keys
    will simply be skipped, with the assumption that the user does not care about that data down-stream. Any data not
    explicitly handled by a slicer here will be passed back based on its value in `original_data` or else the first
    occurrence in the `data` list, with any differing values in subsequent mini-batches being ignored.

    Args:
        slicers: The slicers to use when re-combining the mini-batches
        data: A list of mini-batches.
        original_data: The pre-sliced data. Used as a fallback when slicers don't handle un-slicing things.

    Returns:
        A single combined batch of data.
    """
    if not data:
        return {}
    batch = {}
    processed_keys = set()
    for slicer in slicers:
        for key in slicer.unslice_inputs:
            if key not in data[0]:
                continue
            slices = tuple([minibatch[key] for minibatch in data])
            batch[key] = slicer._unslice_batch(slices, key=key)
            processed_keys.add(key)
    leftover_data = {}
    for key in data[0].keys() - processed_keys:
        original_sample = original_data.get(key, data[0][key])
        if isinstance(original_sample, DistributedValues):
            if original_sample.values[0].shape.rank == 0:
                original_sample = tf.reduce_mean(tuple(d for d in original_sample.values if not tf.math.is_nan(d)))
            else:
                original_sample = tf.concat(original_sample.values, axis=0)
        leftover_data[key] = original_sample
    batch.update(leftover_data)
    return batch
