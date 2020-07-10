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
from typing import Any, ChainMap, Dict, List, MutableMapping, Optional


class Data(ChainMap[str, Any]):
    """A class which contains prediction and batch data.

    This class is intentionally not @traceable.

    Data objects can be interacted with as if they are regular dictionaries. They are however, actually a combination of
    two dictionaries, a dictionary for trace communication and a dictionary of prediction+batch data. In general, data
    written into the trace dictionary will be logged by the system, whereas data in the pred+batch dictionary will not.
    We therefore provide helper methods to write entries into `Data` which are intended or not intended for logging.

    ```python
    d = fe.util.Data({"a":0, "b":1, "c":2})
    a = d["a"]  # 0
    d.write_with_log("d", 3)
    d.write_without_log("e", 5)
    d.write_with_log("a", 4)
    a = d["a"]  # 4
    r = d.read_logs(extra_keys={"c"})  # {"c":2, "d":3, "a":4}
    ```

    Args:
        batch_data: The batch data dictionary. In practice this is itself often a ChainMap containing separate
            prediction and batch dictionaries.
    """
    maps: List[MutableMapping[str, Any]]

    def __init__(self, batch_data: Optional[MutableMapping[str, Any]] = None) -> None:
        super().__init__({}, batch_data or {})

    def write_with_log(self, key: str, value: Any) -> None:
        """Write a given `value` into the `Data` dictionary with the intent that it be logged.

        Args:
            key: The key to associate with the new entry.
            value: The new entry to be written.
        """
        self.__setitem__(key, value)

    def write_without_log(self, key: str, value: Any) -> None:
        """Write a given `value` into the `Data` dictionary with the intent that it not be logged.

        Args:
            key: The ey to associate with the new entry.
            value: The new entry to be written.
        """
        self.maps[1][key] = value

    def read_logs(self) -> Dict[str, Any]:
        """Read all values from the `Data` dictionary which were intended to be logged.

        Returns:
            A dictionary of all of the keys and values to be logged.
        """
        return self.maps[0]
