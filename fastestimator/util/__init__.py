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
                                                                        'NonContext', 'LogSplicer', 'IfElse',
                                                                        'prettify_metric_name', 'strip_suffix',
                                                                        'strip_prefix', 'get_type', 'check_io_names',
                                                                        'filter_nones', 'parse_modes', 'check_ds_id',
                                                                        'is_number', 'DefaultKeyDict', 'FEID', 'Flag',
                                                                        'in_notebook', 'get_shape', 'get_colors',
                                                                        'list_files', 'warn'],
                                                          'util': ['Suppressor', 'Timer', 'cpu_count', 'draw',
                                                                   'get_batch_size', 'get_device', 'get_num_devices',
                                                                   'get_num_gpus', 'pad_batch', 'pad_data',
                                                                   'to_number', 'move_tensors_to_device',
                                                                   'detach_tensors', 'is_valid_file'],
                                                          'cli_util': ['parse_string_to_python'],
                                                          'wget_util': ['bar_custom', 'callback_progress']
                                                          })

if TYPE_CHECKING:
    from fastestimator.util.base_util import FEID, DefaultKeyDict, Flag, IfElse, LogSplicer, NonContext, check_ds_id, \
        check_io_names, filter_nones, get_colors, get_shape, get_type, in_notebook, is_number, list_files, \
        param_to_range, parse_modes, prettify_metric_name, strip_prefix, strip_suffix, to_list, to_set, warn
    from fastestimator.util.cli_util import parse_string_to_python
    from fastestimator.util.data import Data
    from fastestimator.util.img_data import BatchDisplay, GridDisplay, ImageDisplay
    from fastestimator.util.latex_util import AdjustBox, Center, ContainerList, HrefFEID, PyContainer, Verbatim
    from fastestimator.util.traceability_util import FeSplitSummary, trace_model, traceable
    from fastestimator.util.util import Suppressor, Timer, cpu_count, detach_tensors, draw, get_batch_size, \
        get_device, get_num_devices, get_num_gpus, is_valid_file, move_tensors_to_device, pad_batch, pad_data, \
        to_number
    from fastestimator.util.wget_util import bar_custom, callback_progress
