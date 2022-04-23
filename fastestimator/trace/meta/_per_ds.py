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
from typing import Union, List

import functools

from fastestimator.trace.trace import Trace, PerDSTrace
from fastestimator.util.data import Data, DSData
from fastestimator.util.base_util import to_list


def per_ds(clz: type(Trace)):
    """A class annotation which will convert regular traces into dataset-sensitive traces.

    Args:
        clz: The base class to be converted.

    Returns:
        A dataset aware version of the class. Note that if the annotated class instance has a 'per_ds' member variable
        which is set to False, or has outputs containing the '|' character, then a normal (non-ds-aware) instance will
        be returned instead.
    """
    class PerDS(clz, PerDSTrace):
        @functools.wraps(clz.__new__)
        def __new__(cls, *args, **kwargs):
            # We will dynamically determine whether to return a base object or a PerDS variant
            # If any of the outputs already use the | character then we cannot make this a PerDS variant
            base_obj = clz.__new__(clz)
            base_obj.__init__(*args, **kwargs)
            for output in base_obj.outputs:
                if '|' in output:
                    return base_obj
            # If the user set per_ds to False in the constructor then we will not make this a PerDS variant
            if hasattr(base_obj, 'per_ds') and base_obj.per_ds is False:
                return base_obj
            # Otherwise we are good to go with the PerDS variant
            return super().__new__(cls)

        @functools.wraps(clz.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fe_per_ds_trace = clz.__new__(clz)
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
            if self.system.ds_id != '':
                self.fe_per_ds_trace.on_epoch_begin(DSData(self.system.ds_id, data))

        def on_batch_begin(self, data: Data) -> None:
            super().on_batch_begin(data)
            if self.system.ds_id != '':
                self.fe_per_ds_trace.on_batch_begin(DSData(self.system.ds_id, data))

        def on_batch_end(self, data: Data) -> None:
            if self.system.ds_id != '':
                self.fe_per_ds_trace.on_batch_end(DSData(self.system.ds_id, data))
                # Block the main process from writing per-instance info since we already have the more detailed key
                data.per_instance_enabled = False
            super().on_batch_end(data)
            data.per_instance_enabled = True

        def on_ds_end(self, data: Data) -> None:
            if self.system.ds_id != '':
                self.fe_per_ds_trace.on_epoch_end(DSData(self.system.ds_id, data))

        def on_end(self, data: Data) -> None:
            super().on_end(data)
            self.fe_per_ds_trace.on_end(data)

    PerDS.__name__ = clz.__name__
    PerDS.__qualname__ = clz.__qualname__
    PerDS.__module__ = clz.__module__
    PerDS.__doc__ = clz.__doc__  # We want to preserve the docstring of the original class
    return PerDS
