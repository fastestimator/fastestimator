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
from copy import deepcopy
from typing import Optional, Union, Dict, Sequence, Iterable, Any, List

import numpy as np
import tensorflow as tf
from scipy.linalg import hadamard

from fastestimator.dataset.fe_dataset import FEDataset


class LabeledDirDataset(FEDataset):
    """ A dataset which reads files from a folder hierarchy like root/class/data.file

    Args:
        root_dir: The path to the directory containing data sorted by folders
        data_label: What key to assign to the data values in the data dictionary
        key_label: What key to assign to the label values in the data dictionary
        label_mapping: One of * 'string' - The folder names will be used as the labels
                              * 'int' - The folder names will be mapped onto non-negative integers
                              * 'onehot' - The folder names will be mapped onto onehot vectors
                              * 'ecc' - The folder names will be mapped onto an error-correcting-code vector
                              * Dict[string, Any] - A dictionary defining the mapping to use
        file_extension: If provided then only files ending with the file_extension will be included
    """
    def __init__(self,
                 root_dir: str,
                 data_label: str = "x",
                 key_label: str = "y",
                 label_mapping: Union[str, Dict[str, Union[str, int, np.ndarray]]] = "int",
                 file_extension: Optional[str] = None):
        assert isinstance(label_mapping, Dict) or label_mapping in {"string", "int", "onehot", "ecc"}
        data = []
        root_dir = os.path.normpath(root_dir)
        try:
            _, dirs, _ = next(os.walk(root_dir))
            if isinstance(label_mapping, str):
                labels = dirs
                if label_mapping is not "string":
                    num_classes = len(dirs)
                    labels = [i for i in range(num_classes)]
                    if label_mapping is "onehot":
                        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes, dtype='int32')
                    if label_mapping is "ecc":
                        # We'll use a minimum code length of 16, or else whatever power of 2 is >= the number of classes
                        code_length = max(16, 1 << (num_classes - 1).bit_length())
                        labels = hadamard(code_length)
                self.mapping = {clazz: label for clazz, label in zip(dirs, labels)}
            else:
                self.mapping = label_mapping
                assert self.mapping.keys() >= set(dirs), \
                    "Mapping provided to LabeledDirDataset is missing key(s): {}".format(
                        set(dirs) - self.mapping.keys())
            for path, label in self.mapping.items():
                path = os.path.join(root_dir, path)
                _, _, entries = next(os.walk(path))
                data.extend(
                    (os.path.join(path, entry), label) for entry in entries if entry.endswith(file_extension or ""))
        except StopIteration:
            raise ValueError("Invalid directory structure for LabeledDirDataset at root: {}".format(root_dir))
        self.data = {i: {data_label: data[i][0], key_label: data[i][1]} for i in range(len(data))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return deepcopy(self.data[index])

    def get_mapping(self) -> Dict[str, Union[str, int, np.ndarray]]:
        return self.mapping

    @classmethod
    def _skip_init(cls, data: Dict[int, Dict[str, Any]],
                   mapping: Dict[str, Union[str, int, np.ndarray]]) -> 'LabeledDirDataset':
        obj = cls.__new__(cls)
        obj.data = data
        obj.mapping = mapping
        return obj

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['LabeledDirDataset']:
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(LabeledDirDataset._skip_init(data, self.mapping))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        return results


class LabeledDirDatasets:
    """ A class which instantiates multiple LabeledDirDataset from a folder hierarchy like: root/mode/class/data.file

    Args:
        root_dir: The path to the directory containing data sorted by folders
        data_label: What key to assign to the data values in the data dictionary
        key_label: What key to assign to the label values in the data dictionary
        label_mapping: One of * 'string' - The folder names will be used as the labels
                              * 'int' - The folder names will be mapped onto non-negative integers
                              * 'onehot' - The folder names will be mapped onto onehot vectors
                              * 'ecc' - The folder names will be mapped onto an error-correcting-code vector
                              * Dict[string, Any] - A dictionary defining the mapping to use
        file_extension: If provided then only files ending with the file_extension will be included
    """
    def __init__(self,
                 root_dir: str,
                 data_label: str = "x",
                 key_label: str = "y",
                 label_mapping: Union[str, Dict[str, Union[str, int, np.ndarray]]] = "int",
                 file_extension: Optional[str] = None):
        root_dir = os.path.normpath(root_dir)
        try:
            _, dirs, _ = next(os.walk(root_dir))
            self.datasets = {}
            # We're going to do the training data first since it shouldn't have any missing class types for inferring
            # label mapping. Once the label mapping is built it will be passed to all of the other datasets to ensure
            # they use the same values. If 'train' is not available then we just have to pick the first folder and hope
            # that it has all the keys
            first_key = "train" if "train" in dirs else dirs[0]
            self.datasets[first_key] = LabeledDirDataset(root_dir=os.path.join(root_dir, first_key),
                                                         data_label=data_label,
                                                         key_label=key_label,
                                                         label_mapping=label_mapping,
                                                         file_extension=file_extension)
            # This won't change anything if label_mapping was passed in as a dictionary from the start, but will ensure
            # that if we're inferring label mappings we will get consistent mappings over all the data
            label_mapping = self.datasets[first_key].get_mapping()
            self.datasets.update({
                mode: LabeledDirDataset(root_dir=os.path.join(root_dir, mode),
                                        data_label=data_label,
                                        key_label=key_label,
                                        label_mapping=label_mapping,
                                        file_extension=file_extension)
                for mode in dirs if mode is not first_key
            })
        except StopIteration:
            raise ValueError("Invalid directory structure for LabeledDirDatasets at root: {}".format(root_dir))

    def __getitem__(self, mode: str) -> LabeledDirDataset:
        return self.datasets[mode]
