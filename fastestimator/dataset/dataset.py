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
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Union

import jsonpickle
import numpy as np
from torch.utils.data import Dataset

from fastestimator.util.util import get_shape, get_type


class KeySummary:
    """A summary of the dataset attributes corresponding to a particular key.

    Args:
        num_unique_values: The number of unique values corresponding to a particular key (if known).
        shape: The shape of the vectors corresponding to the key. None is used in a list to indicate that a dimension is
            ragged.
        dtype: The data type of instances corresponding to the given key.
    """
    num_unique_values: Optional[int]
    shape: List[Optional[int]]
    dtype: str

    def __init__(self, dtype: str, num_unique_values: Optional[int] = None, shape: List[Optional[int]] = ()) -> None:
        self.num_unique_values = num_unique_values
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return "<KeySummary {}>".format(self.__getstate__())

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DatasetSummary:
    """This class contains information summarizing a dataset object.

    Args:
        num_instances: The number of data instances within the dataset (influences the size of an epoch).
        num_classes: How many different classes are present.
        keys: What keys does the dataset provide, along with summary information about each key.
        class_key: Which key corresponds to class information (if known).
        class_key_mapping: A mapping of the original class string values to the values which are output to the pipeline.
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
        """Defines how many datapoints the dataset contains.

        This is used for computing the number of datapoints available per epoch.

        Returns:
            The number of datapoints within the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch a data instance at a specified index.

        Args:
            index: Which datapoint to retrieve.

        Returns:
            The data dictionary from the specified index.
        """
        raise NotImplementedError

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['FEDataset', List['FEDataset']]:
        """Split this dataset into multiple smaller datasets.

        This function enables several types of splitting:
            1. Splitting by fractions.
                ```python
                ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
                ds2 = ds.split(0.1)  # len(ds) == 900, len(ds2) == 100
                ds3, ds4 = ds.split(0.1, 0.2)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
                ```
            2. Splitting by counts.
                ```python
                ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
                ds2 = ds.split(100)  # len(ds) == 900, len(ds2) == 100
                ds3, ds4 = ds.split(90, 180)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
                ```
            3. Splitting by indices.
                ``python
                ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
                ds2 = ds.split([87,2,3,100,121,158])  # len(ds) == 994, len(ds2) == 6
                ds3 = ds.split(range(100))  # len(ds) == 894, len(ds3) == 100
                ```

        Args:
            *fractions: Floating point values will be interpreted as percentages, integers as an absolute number of
                datapoints, and an iterable of integers as the exact indices of the data that should be removed in order
                to create the new dataset.

        Returns:
            One or more new datasets which are created by removing elements from the current dataset. The number of
            datasets returned will be equal to the number of `fractions` provided. If only a single value is provided
            then the return will be a single dataset rather than a list of datasets.
        """
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
        """The length of a dataset to be used for the purpose of computing splits.

        Useful if sub-classes want to split by something other than indices (see SiameseDirDataset for example).

        Returns:
            The apparent length of the dataset for the purpose of the .split() function
        """
        return len(self)

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['FEDataset']:
        """Split the current dataset apart into several smaller datasets.

        Args:
            splits: Which indices to remove from the current dataset in order to create new dataset(s). One dataset will
                be generated for every iterable within the `splits` sequence.

        Returns:
            New datasets generated by removing data at the indices specified by `splits` from the current dataset.
        """
        raise NotImplementedError

    def summary(self) -> DatasetSummary:
        """Generate a summary representation of this dataset.
        Returns:
            A summary representation of this dataset.
        """
        raise NotImplementedError

    def __str__(self):
        return str(self.summary())


class InMemoryDataset(FEDataset):
    """A dataset abstraction to simplify the implementation of datasets which hold their data in memory.

    Args:
        data: A dictionary like {data_index: {<instance dictionary>}}.
    """
    data: Dict[int, Dict[str, Any]]  # Index-based data dictionary
    summary: lru_cache

    def __init__(self, data: Dict[int, Dict[str, Any]]) -> None:
        self.data = data
        # Normally lru cache annotation is shared over all class instances, so calling cache_clear would reset all
        # caches (for example when calling .split()). Instead we make the lru cache per-instance
        self.summary = lru_cache(maxsize=1)(self.summary)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, str]) -> Union[Dict[str, Any], np.ndarray, List[Any]]:
        """Look up data from the dataset.

        ```python
        data = fe.dataset.InMemoryDataset(...)  # {"x": <100>}, len(data) == 1000
        element = data[0]  # {"x": <100>}
        column = data["x"]  # <1000x100>
        ```

        Args:
            index: Either an int corresponding to a particular element of data, or a string in which case the
                corresponding column of data will be returned.

        Returns:
            A data dictionary if the index was an int, otherwise a column of data in list format.
        """
        if isinstance(index, int):
            return self.data[index]
        else:
            result = [elem[index] for elem in self.data.values()]
            if isinstance(result[0], np.ndarray):
                return np.array(result)
            return result

    def __setitem__(self, key: Union[int, str], value: Union[Dict[str, Any], Sequence[Any]]) -> None:
        """Modify data in the dataset.

        ```python
        data = fe.dataset.InMemoryDataset(...)  # {"x": <100>}, len(data) == 1000
        column = data["x"]  # <1000x100>
        column = column - np.mean(column)
        data["x"] = column
        ```

        Args:
            key: Either an int corresponding to a particular element of data, or a string in which case the
                corresponding column of data will be updated.
            value: The value to be inserted for the given `key`. Must be a dictionary if `key` is an integer. Otherwise
                must be a sequence with the same length as the current length of the dataset.

        Raises:
            AssertionError: If the `value` is inappropriate given the type of the `key`.
        """
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
        """A helper method to create new dataset instances without invoking their __init__ methods.

        Args:
            data: The data dictionary to be used in the new dataset.
            **kwargs: Any other member variables to be assigned in the new dataset.

        Returns:
            A new dataset based on the given inputs.
        """
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
        """Split the current dataset apart into several smaller datasets.

        Args:
            splits: Which indices to remove from the current dataset in order to create new dataset(s). One dataset will
                be generated for every iterable within the `splits` sequence.

        Returns:
            New Datasets generated by removing data at the indices specified by `splits` from the current dataset.
        """
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(self._skip_init(data, **{k: v for k, v in self.__dict__.items() if k not in {'data'}}))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        self.summary.cache_clear()
        return results

    def summary(self) -> DatasetSummary:
        """Generate a summary representation of this dataset.
        Returns:
            A summary representation of this dataset.
        """
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
