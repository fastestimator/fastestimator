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
import warnings
from typing import Any, Dict, Generator, Sequence, Iterable, List, Sized

from fastestimator.dataset.fe_dataset import FEDataset


class GeneratorDataset(FEDataset):
    def __init__(self, generator: Generator[Dict[str, Any], int, None], samples_per_epoch: int):
        self.generator = generator
        self.samples_per_epoch = samples_per_epoch
        next(self.generator)  # Can't send non-none values to a new generator, so need to run a 'warm-up' first

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        return self.generator.send(index)

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['GeneratorDataset']:
        warnings.warn("You probably don't actually want to split a generator dataset")
        results = []
        for split in splits:
            if isinstance(split, Sized):
                size = len(split)
            else:
                # TODO - make this efficient somehow
                size = sum(1 for _ in split)
            results.append(GeneratorDataset(self.generator, size))
            self.samples_per_epoch -= size
        return results
