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
from typing import Any, Union, List

import functools

from fastestimator.trace.trace import Trace, PerDSTrace
from fastestimator.util.data import Data
from fastestimator.util.util import to_list


class DSData(Data):
    # noinspection PyMissingConstructor
    def __init__(self, ds_id: str, data: Data):
        self.maps = data.maps
        self.ds_id = ds_id

    def write_with_log(self, key: str, value: Any) -> None:
        super().write_with_log(key=f'{key}|{self.ds_id}', value=value)

    def write_without_log(self, key: str, value: Any) -> None:
        super().write_without_log(key=f'{key}|{self.ds_id}', value=value)


def per_ds(cls: type(Trace)):
    class PerDS(cls, PerDSTrace):
        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # if self.ds_id is None:
            #     pass  # TODO - maybe only generate extra trace in this case?
            self.fe_per_ds_trace = cls.__new__(cls)
            self.fe_per_ds_trace.__init__(*args, **kwargs)

        def get_outputs(self, ds_ids: Union[None, str, List[str]]) -> List[str]:
            ds_ids = to_list(ds_ids)
            outputs = list(self.outputs)
            for output in self.outputs:
                for ds_id in ds_ids:
                    outputs.append(f"{output}|{ds_id}")
            return outputs

        def on_begin(self, data: Data) -> None:
            super().on_begin(data)
            self.fe_per_ds_trace.on_begin(data)

        def on_ds_begin(self, data: Data) -> None:
            if self.system.ds_id is not None:
                self.fe_per_ds_trace.on_epoch_begin(DSData(self.system.ds_id, data))

        def on_batch_begin(self, data: Data) -> None:
            super().on_batch_begin(data)
            if self.system.ds_id is not None:
                self.fe_per_ds_trace.on_batch_begin(DSData(self.system.ds_id, data))

        def on_batch_end(self, data: Data) -> None:
            super().on_batch_end(data)
            if self.system.ds_id is not None:
                self.fe_per_ds_trace.on_batch_end(DSData(self.system.ds_id, data))

        def on_ds_end(self, data: Data) -> None:
            if self.system.ds_id is not None:
                self.fe_per_ds_trace.on_epoch_end(DSData(self.system.ds_id, data))

        def on_end(self, data: Data) -> None:
            super().on_end(data)
            self.fe_per_ds_trace.on_end(data)

    PerDS.__name__ = cls.__name__
    PerDS.__qualname__ = cls.__qualname__
    PerDS.__module__ = cls.__module__
    return PerDS
