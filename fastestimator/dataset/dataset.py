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
from copy import deepcopy
from typing import Dict, Any, Union, Sequence, Iterable, List

from torch.utils.data import Dataset

from fastestimator.op import get_inputs_by_op, write_outputs_by_op
from fastestimator.op.op import NumpyOp


class FEDataset(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['FEDataset', List['FEDataset']]:
        assert len(fractions) > 0, "split requires at least one fraction argument"
        original_size = len(self)
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
            indices = []
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

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['FEDataset']:
        raise NotImplementedError


class OpDataset(Dataset):
    def __init__(self, dataset: Dataset, ops: List[NumpyOp], mode: str):
        self.dataset = dataset
        self.ops = ops
        self.mode = mode

    def __getitem__(self, index):
        item = deepcopy(self.dataset[index])  # Deepcopy to prevent ops from overwriting values in datasets
        op_data = None
        for op in self.ops:
            op_data = get_inputs_by_op(op, item, op_data)
            op_data = op.forward(op_data, {"mode": self.mode})
            if op.outputs:
                write_outputs_by_op(op, item, op_data)
        return item

    def __len__(self):
        return len(self.dataset)
