# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import wget

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _read_data(file_path: str) -> List[str]:
    with open(file_path, encoding='utf-8') as f:
        data = [x.strip() for x in f if x.strip()]
    return data


def _create_dataset(data_path: str, translate_option: str, extension: str) -> NumpyDataset:
    source, target = translate_option.split("_to_")
    source = source.replace("_", "-")
    if extension != "train":
        source = source.split("-")[0]
    source_data = _read_data(os.path.join(data_path, source + "." + extension))
    target_data = _read_data(os.path.join(data_path, target + "." + extension))
    assert len(target_data) == len(source_data), "Sizes do not match for {} ({} mode)".format(translate_option, extension)
    dataset = NumpyDataset({"source": source_data, "target": target_data})
    return dataset


def load_data(root_dir: Optional[str] = None,
              translate_option: str = "az_to_en") -> Tuple[NumpyDataset, NumpyDataset, NumpyDataset]:
    """Load and return the neural machine translation dataset from TED talks.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        translate_option: Options for translation languages. Available options are: "az_to_en", "az_tr_to_en",
            "be_ru_to_en", "be_to_en", "es_to_pt", "fr_to_pt", "gl_pt_to_en", "gl_to_en", "he_to_pt", "it_to_pt",
            "pt_to_en", "ru_to_en", "ru_to_pt", and "tr_to_en".

    Returns:
        (train_data, eval_data, test_data)
    """
    # Set up path
    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'tednmt')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'tednmt')
    os.makedirs(root_dir, exist_ok=True)
    compressed_path = os.path.join(root_dir, 'qi18naacl-dataset.tar.gz')
    extracted_path = os.path.join(root_dir, 'datasets')
    if not os.path.exists(extracted_path):
        # Download
        if not os.path.exists(compressed_path):
            print("Downloading data to {}".format(compressed_path))
            wget.download('http://www.phontron.com/data/qi18naacl-dataset.tar.gz', compressed_path, bar=bar_custom)
        # Extract
        print("\nExtracting files ...")
        with tarfile.open(compressed_path) as f:
            f.extractall(root_dir)
    # process data
    data_path = os.path.join(extracted_path, translate_option)
    assert os.path.exists(data_path), "folder {} does not exist, please verify translation options".format(data_path)
    train_ds = _create_dataset(data_path=data_path, translate_option=translate_option, extension="train")
    eval_ds = _create_dataset(data_path=data_path, translate_option=translate_option, extension="dev")
    test_ds = _create_dataset(data_path=data_path, translate_option=translate_option, extension="test")
    return train_ds, eval_ds, test_ds
