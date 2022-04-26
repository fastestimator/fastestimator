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
# FEDataset, OpDataset, and FEDataLoader intentionally not imported here to reduce user confusion with auto-complete
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submodules={'data'},
                                            submod_attrs={'batch_dataset': ['BatchDataset'],
                                                          'csv_dataset': ['CSVDataset'],
                                                          'dir_dataset': ['DirDataset'],
                                                          'generator_dataset': ['GeneratorDataset'],
                                                          'labeled_dir_dataset': ['LabeledDirDataset'],
                                                          'numpy_dataset': ['NumpyDataset'],
                                                          'pickle_dataset': ['PickleDataset'],
                                                          'siamese_dir_dataset': ['SiameseDirDataset'],
                                                          'extend_dataset': ['ExtendDataset'], })

if TYPE_CHECKING:
    from fastestimator.dataset import data
    from fastestimator.dataset.batch_dataset import BatchDataset
    from fastestimator.dataset.csv_dataset import CSVDataset
    from fastestimator.dataset.dir_dataset import DirDataset
    from fastestimator.dataset.generator_dataset import GeneratorDataset
    from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset
    from fastestimator.dataset.numpy_dataset import NumpyDataset
    from fastestimator.dataset.pickle_dataset import PickleDataset
    from fastestimator.dataset.siamese_dir_dataset import SiameseDirDataset
    from fastestimator.dataset.extend_dataset import ExtendDataset
