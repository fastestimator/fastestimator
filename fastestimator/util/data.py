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
from typing import Optional, Dict, Any, List, Set, MutableMapping, ChainMap


class Data(ChainMap[str, Any]):
    maps: List[MutableMapping[str, Any]]

    def __init__(self, batch_data: Optional[MutableMapping[str, Any]] = None):
        super().__init__({}, batch_data or {})

    def write_with_log(self, key: str, value: Any):
        self.__setitem__(key, value)

    def write_without_log(self, key: str, value: Any):
        self.maps[1][key] = value

    def read_logs(self, extra_keys: Set[str]) -> Dict[str, Any]:
        return {**{k: v for k, v in self.maps[1].items() if k in extra_keys}, **self.maps[0]}
