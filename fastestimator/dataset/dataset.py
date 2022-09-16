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
from copy import deepcopy
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import jsonpickle
import numpy as np
from torch.utils.data import Dataset

from fastestimator.util.traceability_util import FeSplitSummary, traceable
from fastestimator.util.base_util import get_type, FEID, get_shape


class KeySummary:
    """A summary of the dataset attributes corresponding to a particular key.

    This class is intentionally not @traceable.

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

    This class is intentionally not @traceable.

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


@traceable()
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

    @classmethod
    def fix_split_traceabilty(cls,
                              parent: 'FEDataset',
                              children: List['FEDataset'],
                              fractions: Tuple[Union[float, int, Iterable[int]], ...],
                              seed: Optional[int],
                              stratify: Optional[str]) -> None:
        """A method to fix traceability information after invoking the dataset .split() method.

        Note that the default implementation of the .split() function invokes this already, so this only needs to be
        invoked if you override the .split() method when defining a subclass (ex. BatchDataset).

        Args:
            parent: The parent dataset on which .split() was invoked.
            children: The datasets generated by performing the split.
            fractions: The fraction arguments used to generate the children (should be one-to-one with the children).
            seed: The random seed used to generate the split.
            stratify: The stratify key used to generate the split.
        """
        if hasattr(parent, '_fe_traceability_summary'):
            parent_id = FEID(id(parent))
            fractions = [
                f"range({frac.start}, {frac.stop}, {frac.step})" if isinstance(frac, range) else f"{frac}"
                for frac in fractions
            ]
            for child, frac in zip(children, fractions):
                # noinspection PyProtectedMember
                tables = deepcopy(child._fe_traceability_summary)
                # Update the ID if necessary
                child_id = FEID(id(child))
                if child_id not in tables:
                    # The child was created without invoking its __init__ method, so its internal summary will have the
                    # wrong id
                    table = tables.pop(parent_id)
                    table.fe_id = child_id
                    tables[child_id] = table
                else:
                    table = tables[child_id]
                split_summary = table.fields.get('split', FeSplitSummary())
                split_summary.add_split(parent=parent_id, fraction=frac, seed=seed, stratify=stratify)
                table.fields['split'] = split_summary
                child._fe_traceability_summary = tables
            # noinspection PyUnresolvedReferences
            table = parent._fe_traceability_summary.get(parent_id)
            split_summary = table.fields.get('split', FeSplitSummary())
            split_summary.add_split(parent='self',
                                    fraction=", ".join([f"-{frac}" for frac in fractions]),
                                    seed=seed,
                                    stratify=stratify)
            table.fields['split'] = split_summary
            # Put the new parent summary into the child table to ensure it will always exist in the final set of tables
            for child in children:
                child._fe_traceability_summary[parent_id] = deepcopy(table)

    def split(self,
              *fractions: Union[float, int, Iterable[int]],
              seed: Optional[int] = None,
              stratify: Optional[str] = None) -> Union['FEDataset', List['FEDataset']]:
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
            ```python
            ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
            ds2 = ds.split([87,2,3,100,121,158])  # len(ds) == 994, len(ds2) == 6
            ds3 = ds.split(range(100))  # len(ds) == 894, len(ds3) == 100
            ```

        Args:
            *fractions: Floating point values will be interpreted as percentages, integers as an absolute number of
                datapoints, and an iterable of integers as the exact indices of the data that should be removed in order
                to create the new dataset.
            seed: The random seed to use when splitting the dataset. Useful if you want consistent splits across
                multiple experiments. This isn't necessary if you are splitting by data index.
            stratify: A class key within the dataset with which to stratify the split (to approximately maintain class
                balance ratios before and after a split). Incompatible with data index splitting.

        Returns:
            One or more new datasets which are created by removing elements from the current dataset. The number of
            datasets returned will be equal to the number of `fractions` provided. If only a single value is provided
            then the return will be a single dataset rather than a list of datasets.

        Raises:
            AssertionError: If input arguments are unacceptable.
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

        if method == 'number':
            if stratify is not None:
                splits = self._get_stratified_splits(n_samples, seed, stratify)
            else:
                splits = self._get_fractional_splits(n_samples, seed)
        else:  # method == 'indices':
            assert stratify is None, "Stratify may only be specified when splitting by count or fraction, not by index"
            splits = fractions
        splits = self._do_split(splits)
        FEDataset.fix_split_traceabilty(self, splits, fractions, seed, stratify)
        if len(fractions) == 1:
            return splits[0]
        return splits

    def _get_stratified_splits(self, split_counts: List[int], seed: Optional[int],
                               stratify: str) -> Sequence[Iterable[int]]:
        """Get sequence(s) of indices to split from the current dataset in order to generate new dataset(s).

        Args:
            split_counts: How many datapoints to include in each split.
            seed: What random seed, if any, to use when generating the split(s).
            stratify: A class key within the dataset with which to stratify the split (to approximately maintain class
                balance ratios before and after a split).

        Returns:
            Which data indices to include in each split of data. len(return[i]) == split_counts[i].
        """
        splits = []
        original_size = self._split_length()
        seed_offset = 0

        # Compute the distribution over the stratify key
        distribution = defaultdict(list)
        for idx in range(original_size):
            sample = self[idx]
            key = sample[stratify]
            if hasattr(key, "tobytes"):
                key = key.tobytes()  # Makes numpy arrays hashable
            distribution[key].append(idx)

        supply = {key: len(values) for key, values in distribution.items()}
        split_requests = [{key: (n_split * n_tot) / original_size
                           for key, n_tot in supply.items()} for n_split in split_counts]

        def transfer(source: Dict[Any, int], sink: Dict[Any, int], key: Any, request: int) -> int:
            allowance = min(request, source[key])
            source[key] -= allowance
            sink[key] += allowance
            return allowance

        # Sample splits proportional to the computed distribution
        for split_request, target in zip(split_requests, split_counts):
            split_actual = defaultdict(lambda: 0)
            total = 0
            # Step 1: Try to get as close to the target distribution as possible
            for key, request in split_request.items():
                request = 1 if 0 < request < 1 else round(request)  # Always want at least 1 sample from a class
                total += transfer(source=supply, sink=split_actual, key=key, request=request)
            # Step 2: Correct the error in the total count at the expense of an optimal distribution
            # |total - target| may be > n_keys due to rounding + supply shortage
            spare_last = True  # If we have drawn too many things, we will try not to reduce any classes beneath 1
            while total != target:
                old_total = total
                # Repeatedly add or shave 1 off of everything until we get the correct target number
                for key, requested in sorted(split_actual.items(), key=lambda x: x[1], reverse=True):
                    # reversed to start with the most abundant class
                    if total < target:
                        total += transfer(source=supply, sink=split_actual, key=key, request=1)
                    elif total > target and (not spare_last or requested > 1):
                        total -= transfer(source=split_actual, sink=supply, key=key, request=1)
                    if total == target:
                        break
                if old_total == total:
                    assert spare_last is True, "Cannot stratify the requested split. Please file a bug report."
                    # We weren't able to modify anything, so no choice but to reduce a 1-sample class to 0-sample
                    spare_last = False
            # Step 3: Perform the actual sampling
            split_indices = []
            for key, n_samples in split_actual.items():
                # Dicts have preserved insertion order since python 3.6, so we can increase the seed as we use it
                # to prevent any unintended patterns from emerging while still having consistency over multiple runs.
                # This wouldn't work if the order of encounter of a class can change (like in a generator), but in such
                # cases consistency would be impossible anyways.
                if seed is not None:
                    indices = random.Random(seed + seed_offset).sample(distribution[key], n_samples)
                    seed_offset += 1  # We'll use a different seed each time
                else:
                    indices = random.sample(distribution[key], n_samples)
                split_indices.extend(indices)
                # Sort to allow deterministic seed to work alongside sets
                distribution[key] = sorted(list(set(distribution[key]) - set(indices)))
            if seed is not None:
                random.Random(seed + seed_offset).shuffle(split_indices)
                seed_offset += 1
            else:
                random.shuffle(split_indices)
            splits.append(split_indices)
        return splits

    def _get_fractional_splits(self, split_counts: List[int], seed: Optional[int]) -> Sequence[Iterable[int]]:
        """Get sequence(s) of indices to split from the current dataset in order to generate new dataset(s).

        Args:
            split_counts: How many datapoints to include in each split.
            seed: What random seed, if any, to use when generating the split(s).

        Returns:
            Which data indices to include in each split of data. len(return[i]) == split_counts[i].
        """
        splits = []
        original_size = self._split_length()
        int_sum = sum(split_counts)
        # TODO - convert to a linear congruential generator for large datasets?
        # https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates
        if seed is not None:
            indices = random.Random(seed).sample(range(original_size), int_sum)
        else:
            indices = random.sample(range(original_size), int_sum)
        start = 0
        for stop in split_counts:
            splits.append((indices[i] for i in range(start, start + stop)))
            start += stop
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


@traceable(blacklist=('data', '_summary'))
class InMemoryDataset(FEDataset):
    """A dataset abstraction to simplify the implementation of datasets which hold their data in memory.

    Args:
        data: A dictionary like {data_index: {<instance dictionary>}}.
    """
    data: Dict[int, Dict[str, Any]]  # Index-based data dictionary

    def __init__(self, data: Dict[int, Dict[str, Any]]) -> None:
        self.data = data
        self._summary = None

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
            if index >= len(self):
                raise StopIteration
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
        self._summary = None

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
            obj.__setattr__(k, v)
        obj._summary = None
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
        self._summary = None
        return results

    def summary(self) -> DatasetSummary:
        """Generate a summary representation of this dataset.
        Returns:
            A summary representation of this dataset.
        """
        if self._summary is not None:
            return self._summary
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
        self._summary = DatasetSummary(num_instances=len(self), keys=key_summary)
        return self._summary
