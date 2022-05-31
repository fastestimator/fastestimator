# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import json
import os
from collections import defaultdict
from typing import Any, Optional, Sequence, Union

from natsort import humansorted

from fastestimator.search.search import Search
from fastestimator.summary.summary import ValWithError
from fastestimator.util.base_util import to_set


def _load_search_file(path: str) -> Search:
    path = os.path.abspath(os.path.normpath(path))
    if not os.path.exists(path):
        raise ValueError(f"No file found at location: {path}")
    with open(path, 'r') as file:
        state = json.load(file)
    search = Search.__new__(Search)
    search.name = 'Search'
    search.best_mode = None
    search.optimize_field = None
    search._initialize_state()
    search.__dict__.update(state)
    return search


class SearchData:
    def __init__(self, search: Search, ignore_keys: Union[None, str, Sequence[str]] = None):
        self.params = []
        self.results = []
        self.data = defaultdict(list)
        self.categorical_maps = {}
        self.ignored_params = False  # Did you ignore a param other than search_idx which had more than 1 value?

        search = search.get_search_summary()
        if not search:
            return

        example_item = search[0]
        self.params = set(example_item['param'].keys())
        self.results = set(example_item['result'].keys())

        ignore_keys = to_set(ignore_keys) | {'search_idx'}
        for key in ignore_keys:
            self.params.discard(key)
            self.results.discard(key)

        # Keep a sample parameter value to catch boring parameters
        param_samples = {}

        for elem in search:
            pars = elem['param']
            for k, v in pars.items():
                if k in ignore_keys:
                    if k != 'search_idx' and v != param_samples.setdefault(k, v):
                        # The ignored key had more than 1 value
                        self.ignored_params = True
                    continue
                if k not in self.params:
                    raise ValueError("Inconsistent parameter list detected")
                v = self._parse_value(v)
                param_samples.setdefault(k, v)
                self.data[k].append(v)
            res = elem['result']
            for k, v in res.items():
                if k in ignore_keys:
                    continue
                if k not in self.results:
                    raise ValueError("Inconsistent result list detected")
                self.data[k].append(self._parse_value(v))

        # Remove any parameters which have only 1 value since they are boring to visualize
        for param in list(self.params):  # Copy to a list since it may be modified during iteration
            for val in self.data[param]:
                if val != param_samples[param]:
                    break
            else:
                self.params.discard(param)
                self.data.pop(param)

        # Handle categorical data
        for key, values in self.data.items():
            if all([isinstance(value, (int, float)) for value in values]):
                continue  # Numeric value
            # Else categorical
            categories = humansorted(set(values), reverse=True)
            self.categorical_maps[key] = {cat: i for i, cat in enumerate(categories)}
            self.data[key] = [self.categorical_maps[key][val] for val in values]

        self.params = humansorted(self.params)
        self.results = humansorted(self.results)
        self.inverse_maps = {key: {v2: k2 for k2, v2 in val.items()} for key, val in self.categorical_maps.items()}

    def to_category(self, key: str, val: Any) -> Optional[str]:
        m = self.inverse_maps.get(key, {})
        if not m:
            return val if val is None else str(val)
        return m[val]

    @staticmethod
    def _parse_value(value: Any) -> Union[int, float, str, None]:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
            value = value.item()
        if isinstance(value, ValWithError):
            value = value.y
        if not isinstance(value, (int, float, str, type(None))):
            value = str(value)
        return value
