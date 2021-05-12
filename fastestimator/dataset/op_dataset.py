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
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
from torch.utils.data import Dataset

from fastestimator.dataset import BatchDataset
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import pad_batch


class _DelayedDeepDict(dict):
    """A class to perform delayed deep copying from another dictionary.

    This class is intentionally not @traceable (need quick instantiation).

    This can be useful to reduce memory overhead while protecting the original dictionary entries being overridden. It's
    currently a private class because it doesn't implement the full dictionary spec in an intuitive way.

    Args:
        base: The dictionary to be wrapped.
    """
    def __init__(self, base: Dict[str, Any]):
        super().__init__()
        self.base = base

    def __getitem__(self, key: str) -> Any:
        if key not in self and key in self.base:
            self[key] = deepcopy(self.base[key])
        return super().__getitem__(key)

    def __delitem__(self, key: str):
        # The key will still be in the base dictionary, but that can be handled by the finalize later.
        if key in self or key not in self.base:
            # 'key not in base' to raise errors when key doesn't exist
            super().__delitem__(key)

    def finalize(self, retain: Optional[Set[str]] = None, deep_remainder: bool = True) -> None:
        """Finish migrating the data from the original dictionary into this one.

        Args:
            retain: Which keys to keep (or an empty set to keep all available keys).
            deep_remainder: Whether to deep copy any keys which have not yet been copied.
        """
        for key in self.base:
            if retain and key not in retain:
                continue
            if key not in self:
                if deep_remainder:
                    self[key] = deepcopy(self.base[key])
                else:
                    self[key] = self.base[key]
        if retain:
            for key in self.keys() - retain:
                del self[key]
        self.base = {}


@traceable()
class OpDataset(Dataset):
    """A wrapper for datasets which allows operators to be applied to them in a pipeline.

    This class should not be directly instantiated by the end user. The fe.Pipeline will automatically wrap datasets
    within an Op dataset as needed.

    Args:
        dataset: The base dataset to wrap.
        ops: A list of ops to be applied after the base `dataset` `__getitem__` is invoked.
        mode: What mode the system is currently running in ('train', 'eval', 'test', or 'infer').
        output_keys: What keys can be produced from pipeline. If None, all keys will be considered.
        deep_remainder: Whether data which is not modified by Ops should be deep copied or not. This argument is used to
            help with RAM management, but end users can almost certainly ignore it.
    """
    def __init__(self,
                 dataset: Dataset,
                 ops: List[NumpyOp],
                 mode: str,
                 output_keys: Optional[Set[str]] = None,
                 deep_remainder: bool = True) -> None:
        self.dataset = dataset
        if isinstance(self.dataset, BatchDataset):
            self.dataset.reset_index_maps()
        self.ops = ops
        self.mode = mode
        self.output_keys = output_keys
        self.deep_remainder = deep_remainder

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Fetch a data instance at a specified index, and apply transformations to it.

        Args:
            index: Which datapoint to retrieve.

        Returns:
            The data dictionary from the specified index, with transformations applied.
        """
        if isinstance(self.dataset, BatchDataset):
            # BatchDataset may randomly sample the same elements multiple times, so need to avoid reprocessing
            unique_samples = {}  # id: idx
            results = []
            for idx, data in enumerate(self.dataset[index]):
                data_id = id(data)
                if data_id not in unique_samples:
                    data = _DelayedDeepDict(data)
                    forward_numpyop(self.ops, data, {'mode': self.mode})
                    data.finalize(retain=self.output_keys, deep_remainder=self.deep_remainder)
                    results.append(data)
                    unique_samples[data_id] = idx
                else:
                    results.append(results[unique_samples[data_id]])
            if self.dataset.pad_value is not None:
                pad_batch(results, self.dataset.pad_value)
            results = {key: np.array([item[key] for item in results]) for key in results[0]}
        else:
            results = _DelayedDeepDict(self.dataset[index])
            forward_numpyop(self.ops, results, {'mode': self.mode})
            results.finalize(retain=self.output_keys, deep_remainder=self.deep_remainder)
        return results

    def __len__(self):
        return len(self.dataset)
