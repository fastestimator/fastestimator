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
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submod_attrs={'data': ['Data'],
                                                          'img_data': ['ImageDisplay', 'BatchDisplay', 'GridDisplay'],
                                                          'latex_util': ['AdjustBox', 'Center', 'ContainerList',
                                                                         'HrefFEID', 'PyContainer', 'Verbatim'],
                                                          'traceability_util': ['FeSplitSummary', 'trace_model',
                                                                                'traceable'],
                                                          'base_util': ['to_set', 'to_list', 'param_to_range',
                                                                        'NonContext', 'Suppressor', 'LogSplicer',
                                                                        'prettify_metric_name', 'strip_suffix',
                                                                        'strip_prefix', 'get_type', 'check_io_names',
                                                                        'parse_modes', 'check_ds_id', 'is_number',
                                                                        'DefaultKeyDict', 'FEID', 'Flag', 'in_notebook',
                                                                        'get_shape', 'get_colors', 'list_files'],
                                                          'util': ['Timer', 'cpu_count', 'draw', 'get_batch_size',
                                                                   'get_num_devices', 'pad_batch', 'pad_data',
                                                                   'to_number'],
                                                          'cli_util': ['parse_string_to_python'],
                                                          'wget_util': ['bar_custom', 'callback_progress']
                                                          })

if TYPE_CHECKING:
    from fastestimator.util.data import Data
    from fastestimator.util.img_data import ImageDisplay, BatchDisplay, GridDisplay
    from fastestimator.util.latex_util import AdjustBox, Center, ContainerList, HrefFEID, PyContainer, Verbatim
    from fastestimator.util.traceability_util import FeSplitSummary, trace_model, traceable
    from fastestimator.util.base_util import to_set, to_list, param_to_range, NonContext, Suppressor, LogSplicer, \
        prettify_metric_name, strip_suffix, strip_prefix, get_type, check_io_names, parse_modes, check_ds_id, \
        is_number, DefaultKeyDict, FEID, Flag, in_notebook, get_shape, get_colors, list_files
    from fastestimator.util.cli_util import parse_string_to_python
    from fastestimator.util.util import Timer, cpu_count, draw, get_batch_size, get_num_devices, pad_batch, pad_data, \
        to_number
    from fastestimator.util.wget_util import bar_custom, callback_progress
