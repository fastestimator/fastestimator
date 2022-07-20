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
                                            submod_attrs={'best_model_saver': ['BestModelSaver'],
                                                          'csv_logger': ['CSVLogger'],
                                                          'image_saver': ['ImageSaver'],
                                                          'image_viewer': ['ImageViewer'],
                                                          'model_saver': ['ModelSaver'],
                                                          'restore_wizard': ['RestoreWizard'],
                                                          'tensorboard': ['TensorBoard'],
                                                          'test_report': ['TestReport'],
                                                          'traceability': ['Traceability'],
                                                          'batch_display': ['BatchDisplay'],
                                                          'grid_display': ['GridDisplay']})

if TYPE_CHECKING:
    from fastestimator.trace.io.batch_display import BatchDisplay
    from fastestimator.trace.io.best_model_saver import BestModelSaver
    from fastestimator.trace.io.csv_logger import CSVLogger
    from fastestimator.trace.io.grid_display import GridDisplay
    from fastestimator.trace.io.image_saver import ImageSaver
    from fastestimator.trace.io.image_viewer import ImageViewer
    from fastestimator.trace.io.model_saver import ModelSaver
    from fastestimator.trace.io.restore_wizard import RestoreWizard
    from fastestimator.trace.io.tensorboard import TensorBoard
    from fastestimator.trace.io.test_report import TestReport
    from fastestimator.trace.io.traceability import Traceability
