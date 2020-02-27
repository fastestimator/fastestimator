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

from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple, Set, Sequence, Iterable

import numpy as np

from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset


class SiameseDirDataset(LabeledDirDataset):
    """ A dataset which reads files from a folder hierarchy like root/class(/es)/data.file. Data is returned in pairs, 
        where the label value is 1 if the data are drawn from the same class, and 0 otherwise. One epoch is defined as 
        the time it takes to visit every data point exactly once as the 'data_key_left'. Each data point may occur zero
        or many times as 'data_key_right' within the same epoch. SiameseDirDataset.split() will split by class index 
        rather than by data instance index. 

    Args:
        root_dir: The path to the directory containing data sorted by folders
        data_key_left: What key to assign to the first data element in the pair
        data_key_right: What key to assign to the second data element in the pair
        label_key: What key to assign to the label values in the data dictionary
        percent_matching_data: What percentage of the time should data be paired by class (label value = 1)
        label_mapping: A dictionary defining the mapping to use. If not provided will map classes to int labels
        file_extension: If provided then only files ending with the file_extension will be included
    """

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

    def _split_length(self) -> int:
        return len(self.class_data)

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['SiameseDirDataset']:
        # Splits in this context refer to class indices rather than the typical data indices
        results = []
        for split in splits:
            # Convert class indices to data indices
            int_class_keys = list(sorted(self.class_data.keys()))
            split = [item for i in split for item in self.class_data[int_class_keys[i]]]
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
