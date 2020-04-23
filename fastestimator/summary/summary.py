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
from collections import defaultdict
from typing import Optional


class Summary:
    """A summary object that records training history.

    Args:
        name: Name of the experiment. If None then experiment results will be ignored.
    """
    def __init__(self, name: Optional[str]) -> None:
        self.name = name
        self.history = defaultdict(lambda: defaultdict(dict))  # {mode: {key: {step: value}}}

    def merge(self, other: 'Summary'):
        """Merge another `Summary` into this one.

        Args:
            other: Other `summary` object to be merged.
        """
        for mode, sub in other.history.items():
            for key, val in sub.items():
                self.history[mode][key].update(val)

    def __bool__(self) -> bool:
        """Whether training history should be recorded.

        Returns:
            True iff this `Summary` has a non-None name.
        """
        return bool(self.name)
