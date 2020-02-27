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
# FEDataset and OpDataset intentionally not imported here to reduce user confusion with auto-complete
from fastestimator.dataset.csv_dataset import CSVDataset
from fastestimator.dataset.data import usps, montgomery, mnist, cifar10, nih_chestxray, horse2zebra, cub200, mendeley, \
    svhn, mscoco, omniglot
from fastestimator.dataset.generator_dataset import GeneratorDataset
from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.dataset.unlabeled_dir_dataset import UnlabeledDirDataset
from fastestimator.dataset.unpaired_dataset import UnpairedDataset
from fastestimator.dataset.pickle_dataset import PickleDataset
from fastestimator.dataset.siamese_dir_dataset import SiameseDirDataset
