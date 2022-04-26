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
                                            submod_attrs={'logs': ['configure_log_parser'],
                                                          'main': ['run_main'],
                                                          'plot': ['configure_plot_parser'],
                                                          'run': ['configure_run_parser'],
                                                          'train': ['configure_test_parser', 'configure_train_parser']})

if TYPE_CHECKING:
    from fastestimator.cli.logs import configure_log_parser
    from fastestimator.cli.main import run_main
    from fastestimator.cli.plot import configure_plot_parser
    from fastestimator.cli.run import configure_run_parser
    from fastestimator.cli.train import configure_test_parser, configure_train_parser
