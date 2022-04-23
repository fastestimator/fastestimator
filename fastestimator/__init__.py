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
import inspect
import re
import sys
from typing import TYPE_CHECKING

if sys.platform == 'darwin':
    # Block the tkinter module from being imported on Mac. This is necessary in order for Mac multiprocessing to work,
    # since other modules such as nltk import tkinter, and it seems more likely that AI developers will need
    # multiprocessing than tkinter.
    sys.modules['tkinter'] = None

# Fix known bugs with libraries which use multiprocessing in a way which conflicts with pytorch data loader
import cv2
cv2.setNumThreads(0)
try:
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
except ModuleNotFoundError:
    pass

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submodules={'architecture', 'backend', 'dataset', 'layers', 'op',
                                                        'schedule', 'search', 'summary', 'trace', 'util', 'xai'},
                                            submod_attrs={'estimator': ['Estimator', 'enable_deterministic',
                                                                        'record_history'],
                                                          'network': ['Network', 'build'],
                                                          'pipeline': ['Pipeline']})

if TYPE_CHECKING:
    # Allow IDEs to play nice with lazy loading
    from fastestimator import architecture, backend, dataset, layers, op, schedule, search, summary, trace, util, xai
    from fastestimator.estimator import Estimator, enable_deterministic, record_history
    from fastestimator.network import Network, build
    from fastestimator.pipeline import Pipeline

__version__ = '1.5.0'
fe_deterministic_seed = None
fe_history_path = None  # Where to save training histories. None for ~/fastestimator_data/history.db, False to disable
fe_build_count = 0

# Disable history logging for tests by default (they can still turn it on/off manually in setUpClass/tearDownClass)
if __name__ != '__main__':
    for frame in inspect.stack()[1:]:
        if frame.filename[0] != '<':  # Filenames starting with '<' are internal
            # frame.filename will be the name of the file which is currently importing FE
            if re.match('.*/test/PR_test/.*test_.*\\.py', frame.filename):
                fe_history_path = False
            break
