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
from copy import deepcopy
from multiprocessing import Lock
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import numpy as np
from torch.utils.data import Dataset

from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.util.data import FilteredData
from fastestimator.util.traceability_util import traceable


class _DelayedDeepDict(dict):
    """A class to perform delayed deep copying from another dictionary.

    This class is intentionally not @traceable (need quick instantiation).

    This can be useful to reduce memory overhead while protecting the original dictionary entries being overridden. It's
    currently a private class because it doesn't implement the full dictionary spec in an intuitive way.

    Args:
        base: The dictionary to be wrapped.
    """
    to_warn: Set[str] = set()

    def __init__(self, base: Dict[str, Any]):
        super().__init__()
        self.base = base
        self.mask = set()
        self.warn = False

    def __getitem__(self, key: str) -> Any:
        if key not in self and key in self.base and key not in self.mask:
            item = self.base[key]
            if isinstance(item, np.ndarray):
                # We'll avoid copying ndarrays for speed/memory savings. The forward_numpyop function will copy later if
                # required.
                item = item.view()
                item.flags.writeable = False
                self[key] = item
            else:
                self[key] = deepcopy(item)
        return super().__getitem__(key)

    def __delitem__(self, key: str):
        # The key will still be in the base dictionary, but that can be handled by the finalize later.
        if key in self or key not in self.base:
            # 'key not in base' to raise errors when key doesn't exist
            super().__delitem__(key)
        self.mask.add(key)  # Mask the key so that it cannot be drawn out of the base

    def finalize(self, retain: Optional[Set[str]] = None, deep_remainder: bool = True) -> None:
        """Finish migrating the data from the original dictionary into this one.

        Args:
            retain: Which keys to keep (or an empty set to keep all available keys).
            deep_remainder: Whether to deep copy any keys which have not yet been copied.
        """
        for key in self.base:
            if key in self.mask:
                continue
            if retain and key not in retain:
                if key not in self.to_warn:
                    self.to_warn.add(key)
                    self.warn = True
                continue
            if key not in self:
                if deep_remainder:
                    self[key] = deepcopy(self.base[key])
                else:
                    self[key] = self.base[key]
        if retain:
            for key in self.keys() - retain:
                if key not in self.to_warn:
                    self.to_warn.add(key)
                    self.warn = True
                del self[key]
        self.base = {}
        # We need to mark all of the arrays as writeable again to avoid warnings from pytorch. Once torch wraps the
        # arrays, in-place edits on the torch tensors do not impact the numpy arrays anyways.
        for val in self.values():
            if isinstance(val, np.ndarray):
                val.flags.writeable = True


@traceable(blacklist='lock')
class OpDataset(Dataset):
    """A wrapper for datasets which allows operators to be applied to them in a pipeline.

    This class should not be directly instantiated by the end user. The fe.Pipeline will automatically wrap datasets
    within an Op dataset as needed.

    Args:
        dataset: The base dataset to wrap.
        ops: A list of ops to be applied after the base `dataset` `__getitem__` is invoked.
        mode: What mode the system is currently running in ('train', 'eval', 'test', or 'infer').
        output_keys: What keys can be produced from pipeline. If None or empty, all keys will be considered.
        deep_remainder: Whether data which is not modified by Ops should be deep copied or not. This argument is used to
            help with RAM management, but end users can almost certainly ignore it.
    """
    to_warn: Set[str] = set()
    warned: Set[str] = set()

    def __init__(self,
                 dataset: Dataset,
                 ops: List[NumpyOp],
                 mode: str,
                 output_keys: Optional[Set[str]] = None,
                 deep_remainder: bool = True) -> None:
        # Track whether this dataset returns batches or not (useful for pipeline and traceability)
        if not hasattr(dataset, "fe_batch"):
            sample_item = dataset[0]
            dataset.fe_batch = len(sample_item) if isinstance(sample_item, list) else 0
        self.dataset = dataset
        self.fe_batch = dataset.fe_batch
        if hasattr(dataset, "fe_reset_ds"):
            self.fe_reset_ds = dataset.fe_reset_ds
        if hasattr(dataset, "fe_batch_indices"):
            self.fe_batch_indices = dataset.fe_batch_indices
        self.ops = ops
        self.mode = mode
        self.output_keys = output_keys
        self.deep_remainder = deep_remainder
        self.lock = Lock()

    def __getitem__(self, index: int) -> Union[Mapping[str, Any], List[Mapping[str, Any]], FilteredData]:
        """Fetch a data instance at a specified index, and apply transformations to it.

        Args:
            index: Which datapoint to retrieve.

        Returns:
            The data dictionary from the specified index, with transformations applied OR an indication that this index
            should be thrown out.
        """
        item = self.dataset[index]
        if isinstance(item, list):
            # BatchDataset may randomly sample the same elements multiple times, so need to avoid reprocessing
            unique_samples = {}  # id: idx
            results = []
            for idx, data in enumerate(item):
                data_id = id(data)
                if data_id not in unique_samples:
                    data = _DelayedDeepDict(data)
                    filter_data = forward_numpyop(self.ops, data, {'mode': self.mode})
                    if filter_data:
                        results.append(filter_data)
                    else:
                        data.finalize(retain=self.output_keys, deep_remainder=self.deep_remainder)
                        results.append(data)
                        if data.warn:
                            self.to_warn |= (data.to_warn - self.warned)
                    unique_samples[data_id] = idx
                else:
                    results.append(results[unique_samples[data_id]])
        else:
            results = _DelayedDeepDict(item)
            filter_data = forward_numpyop(self.ops, results, {'mode': self.mode})
            if filter_data:
                return filter_data
            results.finalize(retain=self.output_keys, deep_remainder=self.deep_remainder)
            if results.warn:
                self.to_warn |= (results.to_warn - self.warned)
        if self.to_warn and self.lock.acquire(block=False):
            self.warned.update(self.to_warn)
            print("FastEstimator-Warn: The following key(s) are being pruned since they are unused outside of the "
                  "Pipeline. To prevent this, you can declare the key(s) as inputs to Traces or TensorOps: "
                  f"{', '.join(self.to_warn)}")
            self.to_warn.clear()
            # We intentionally never release the lock so that during multi-threading only 1 message can be printed
        return results

    def __len__(self):
        return len(self.dataset)
