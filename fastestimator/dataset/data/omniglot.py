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

import os
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set, Sequence, Iterable

import numpy as np
import wget

from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


class SiameseDirDataset(LabeledDirDataset):
    class_data: Dict[Any, Set[int]]

    def __init__(self,
                 root_dir: str,
                 data_key_left: str = "x_a",
                 data_key_right: str = "x_b",
                 label_key: str = "y",
                 percent_matching_data: float = 0.5,
                 label_mapping: Optional[Dict[str, Any]] = None,
                 file_extension: Optional[str] = None):
        super().__init__(root_dir, data_key_left, label_key, label_mapping, file_extension)
        self.class_data = self._data_to_class(self.data, label_key)
        self.percent_matching_data = percent_matching_data
        self.data_key_left = data_key_left
        self.data_key_right = data_key_right
        self.label_key = label_key

    @staticmethod
    def _data_to_class(data: Dict[int, Dict[str, Any]], label_key: str) -> Dict[Any, Set[int]]:
        class_data = {}
        for idx, elem in data.items():
            class_data.setdefault(elem[label_key], set()).add(idx)
        return class_data

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['SiameseDirDataset']:
        # TODO - This dataset should split based on classes / class groupings rather than indices
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            class_data = self._data_to_class(data, self.label_key)
            results.append(
                SiameseDirDataset._skip_init(data,
                                             self.mapping,
                                             class_data=class_data,
                                             percent_matching_data=self.percent_matching_data,
                                             data_key_left=self.data_key_left,
                                             data_key_right=self.data_key_right,
                                             label_key=self.label_key))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        self.class_data = self._data_to_class(self.data, self.label_key)
        return results

    def __getitem__(self, index: int):
        base_item = deepcopy(self.data[index])
        if np.random.uniform(0, 1) < self.percent_matching_data:
            # Generate matching data
            clazz_items = self.class_data[base_item[self.label_key]]
            other = np.random.choice(list(clazz_items - {index}))
            base_item[self.data_key_right] = self.data[other][self.data_key_left]
            base_item[self.label_key] = 1
        else:
            # Generate non-matching data
            other_classes = self.class_data.keys() - {base_item[self.label_key]}
            other_class = np.random.choice(list(other_classes))
            other = np.random.choice(list(self.class_data[other_class]))
            base_item[self.data_key_right] = self.data[other][self.data_key_left]
            base_item[self.label_key] = 0
        return base_item

    def one_shot_trial(self, n: int) -> Tuple[List[str], List[str]]:
        """
        Generate one-shot trial data, where the similarity should be highest between the index 0 elements of the arrays
        
        Args:
            n: The number of samples to draw for computing one shot accuracy. Should be <= the number of total classes

        Returns:
            ([class_a_instance_x, class_a_instance_x, class_a_instance_x, ...], 
            [class_a_instance_w, class_b_instance_y, class_c_instance_z, ...])
        """
        assert n > 1, "one_shot_trial requires an n-value of at least 2"
        assert n <= len(self.class_data.keys()), \
            "one_shot_trial only supports up to {} comparisons, but an n-value of {} was given".format(
                len(self.class_data.keys()), n)
        classes = np.random.choice(list(self.class_data.keys()), size=n, replace=False)
        base_image_indices = np.random.choice(list(self.class_data[classes[0]]), size=2, replace=False)
        l1 = [self.data[base_image_indices[0]][self.data_key_left]] * n
        l2 = [self.data[base_image_indices[1]][self.data_key_left]]
        for clazz in classes[1:]:
            index = np.random.choice(list(self.class_data[clazz]))
            l2.append(self.data[index][self.data_key_left])
        return l1, l2


def load_data(root_dir: Optional[str] = None) -> Tuple[SiameseDirDataset, SiameseDirDataset]:
    """Download the Omniglot dataset to local storage.
    Args:
        root_dir: The path to store the  data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.
    Returns:
        TrainData, EvalData
    """
    if root_dir is None:
        root_dir = os.path.join(str(Path.home()), 'fastestimator_data', 'Omniglot')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'Omniglot')
    os.makedirs(root_dir, exist_ok=True)

    train_path = os.path.join(root_dir, 'images_background')
    eval_path = os.path.join(root_dir, 'images_evaluation')
    train_zip = os.path.join(root_dir, 'images_background.zip')
    eval_zip = os.path.join(root_dir, 'images_evaluation.zip')

    files = [(train_path, train_zip, 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'),
             (eval_path, eval_zip, 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip')]

    for data_path, data_zip, download_link in files:
        if not os.path.exists(data_path):
            # Download
            if not os.path.exists(data_zip):
                print("Downloading data: {}".format(data_zip))
                wget.download(download_link, data_zip, bar=bar_custom)
            # Extract
            print("Extracting data: {}".format(data_path))
            with zipfile.ZipFile(data_zip, 'r') as zip_file:
                zip_file.extractall(root_dir)

    return SiameseDirDataset(train_path), SiameseDirDataset(eval_path)
