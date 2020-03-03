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
import math
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Any, Union, Sequence, Iterable, List, Optional, Hashable
import numpy as np
import jsonpickle
from torch.utils.data import Dataset

from fastestimator.util.util import get_type, get_shape


class KeySummary:
    """
    A summary of the dataset attributes corresponding to a particular key
    
    Args:
        num_unique_values: The number of unique values corresponding to a particular key (if known)
        shape: The shape of the vectors corresponding to the key. None is used in a list to indicate that a dimension is
            ragged.
        dtype: The data type of instances corresponding to the given key
    """
    num_unique_values: Optional[int]
    shape: List[Optional[int]]
    dtype: str

    def __init__(self, dtype: str, num_unique_values: Optional[int] = None, shape: List[Optional[int]] = ()):
        self.num_unique_values = num_unique_values
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return "<KeySummary {}>".format(self.__getstate__())

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DatasetSummary:
    """
    This class contains information summarizing a dataset object

    Args:
        num_instances: The number of data instances within the dataset (influences the size of an epoch)
        num_classes: How many different classes are there
        keys: What keys does the dataset provide, along with summary information about each key
        class_key: Which key corresponds to class information (if known)
        class_key_mapping: A mapping of the original class string values to the values which are output to the pipeline
    """
    num_instances: int
    num_classes: Optional[int]
    class_key: Optional[str]
    class_key_mapping: Optional[Dict[str, Any]]
    keys: Dict[str, KeySummary]

    def __init__(self,
                 num_instances: int,
                 keys: Dict[str, KeySummary],
                 num_classes: Optional[int] = None,
                 class_key_mapping: Optional[Dict[str, Any]] = None,
                 class_key: Optional[str] = None):
        self.num_instances = num_instances
        self.class_key = class_key
        self.num_classes = num_classes
        self.class_key_mapping = class_key_mapping
        self.keys = keys

    def __repr__(self):
        return "<DatasetSummary {}>".format(self.__getstate__())

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __str__(self):
        return jsonpickle.dumps(self, unpicklable=False)


class FEDataset(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['FEDataset', List['FEDataset']]:
        assert len(fractions) > 0, "split requires at least one fraction argument"
        original_size = self._split_length()
        method = None
        frac_sum = 0
        int_sum = 0
        n_samples = []
        for frac in fractions:
            if isinstance(frac, float):
                frac_sum += frac
                frac = math.ceil(original_size * frac)
                int_sum += frac
                n_samples.append(frac)
                if method is None:
                    method = 'number'
                assert method == 'number', "Split supports either numeric splits or lists of indices but not both"
            elif isinstance(frac, int):
                int_sum += frac
                n_samples.append(frac)
                if method is None:
                    method = 'number'
                assert method == 'number', "Split supports either numeric splits or lists of indices but not both"
            elif isinstance(frac, Iterable):
                if method is None:
                    method = 'indices'
                assert method == 'indices', "Split supports either numeric splits or lists of indices but not both"
            else:
                raise ValueError(
                    "split only accepts float, int, or iter[int] type splits, but {} was given".format(frac))
        assert frac_sum < 1, "total split fraction should sum to less than 1.0, but got: {}".format(frac_sum)
        assert int_sum < original_size, \
            "total split requirements ({}) should sum to less than dataset size ({})".format(int_sum, original_size)

        splits = []
        if method == 'number':
            # TODO - convert to a linear congruential generator for large datasets?
            # https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates
            indices = random.sample(range(original_size), int_sum)
            start = 0
            for stop in n_samples:
                splits.append((indices[i] for i in range(start, start + stop)))
                start += stop
        elif method == 'indices':
            splits = fractions
        splits = self._do_split(splits)
        if len(fractions) == 1:
            return splits[0]
        return splits

    def _split_length(self) -> int:
        # Useful if sub-classes want to split by something other than indices (see SiameseDirDataset for example)
        return len(self)

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['FEDataset']:
        raise NotImplementedError

    def summary(self) -> DatasetSummary:
        raise NotImplementedError

    def __str__(self):
        return str(self.summary())


class InMemoryDataset(FEDataset):
    data: Dict[int, Dict[str, Any]]  # Index-based data dictionary
    summary: lru_cache

    def __init__(self, data: Dict[int, Dict[str, Any]]):
        self.data = data
        # Normally lru cache annotation is shared over all class instances, so calling cache_clear would reset all
        # caches (for example when calling .split()). Instead we make the lru cache per-instance
        self.summary = lru_cache(maxsize=1)(self.summary)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, str]) -> Union[Dict[str, Any], np.ndarray, List[Any]]:
        if isinstance(index, int):
            return self.data[index]
        else:
            result = [elem[index] for elem in self.data.values()]
            if isinstance(result[0], np.ndarray):
                return np.array(result)
            return result

    def __setitem__(self, key: Union[int, str], value: Union[Dict[str, Any], Sequence[Any]]):
        if isinstance(key, int):
            assert isinstance(value, Dict), "if setting a value using an integer index, must provide a dictionary"
            self.data[key] = value
        else:
            assert len(value) == len(self.data), \
                "input value must be of length {}, but had length {}".format(len(self.data), len(value))
            for i in range(len(self.data)):
                self.data[i][key] = value[i]
        self.summary.cache_clear()

    def _skip_init(self, data: Dict[int, Dict[str, Any]], **kwargs) -> 'InMemoryDataset':
        obj = self.__class__.__new__(self.__class__)
        obj.data = data
        for k, v in kwargs.items():
            if k == 'summary':
                continue  # Ignore summary object since we're going to re-initialize it
            else:
                obj.__setattr__(k, v)
        obj.summary = lru_cache(maxsize=1)(obj.summary)
        return obj

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['InMemoryDataset']:
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(self._skip_init(data, **{k: v for k, v in self.__dict__.items() if k not in {'data'}}))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        self.summary.cache_clear()
        return results

    def summary(self) -> DatasetSummary:
        # We will check whether the dataset is doing additional pre-processing on top of the self.data keys. If not we
        # can extract extra information about the data without incurring a large computational time cost
        final_example = self[0]
        original_example = self.data[0]
        keys = final_example.keys()
        shapes = {}
        dtypes = {}
        n_unique_vals = defaultdict(lambda: 0)
        for key in keys:
            final_val = final_example[key]
            # TODO - if val is empty list, should find a sample which has entries
            dtypes[key] = get_type(final_val)
            shapes[key] = get_shape(final_val)

            # Check whether type and shape have changed by get_item
            if key in original_example:
                original_val = original_example[key]
                original_dtype = get_type(original_val)
                original_shape = get_shape(original_val)

                # If no changes, then we can relatively quickly count the unique values using self.data
                if dtypes[key] == original_dtype and shapes[key] == original_shape and isinstance(
                        original_val, Hashable):
                    n_unique_vals[key] = len({self.data[i][key] for i in range(len(self.data))})

        key_summary = {
            key: KeySummary(dtype=dtypes[key], num_unique_values=n_unique_vals[key] or None, shape=shapes[key])
            for key in keys
        }
        return DatasetSummary(num_instances=len(self), keys=key_summary)
