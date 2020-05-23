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
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm

from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset


def load_data(root_dir: Optional[str] = None) -> Tuple[LabeledDirDataset, LabeledDirDataset]:
    """Load and return the Mendeley dataset.

    Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
    Images for Classification", Mendeley Data, v2 http://dx.doi.org/10.17632/rscbjbr9sj.2

    CC BY 4.0 licence:
    https://creativecommons.org/licenses/by/4.0/

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, test_data)
    """
    url = 'https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/' \
          'ChestXRay2017.zip?dl=1'
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'Mendeley')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'Mendeley')
    os.makedirs(root_dir, exist_ok=True)

    data_compressed_path = os.path.join(root_dir, 'ChestXRay2017.zip')
    data_folder_path = os.path.join(root_dir, 'chest_xray')

    if not os.path.exists(data_folder_path):
        # download
        if not os.path.exists(data_compressed_path):
            print("Downloading data to {}".format(root_dir))
            stream = requests.get(url, stream=True)  # python wget does not work
            total_size = int(stream.headers.get('content-length', 0))
            block_size = int(1e6)  # 1 MB
            progress = tqdm(total=total_size, unit='B', unit_scale=True)
            with open(data_compressed_path, 'wb') as outfile:
                for data in stream.iter_content(block_size):
                    progress.update(len(data))
                    outfile.write(data)
            progress.close()

        # extract
        print("\nExtracting file ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            # There's some garbage data from macOS in the zip file that gets filtered out here
            zip_file.extractall(root_dir, filter(lambda x: x.startswith("chest_xray/"), zip_file.namelist()))

    label_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}
    return LabeledDirDataset(os.path.join(data_folder_path, "train"), label_mapping=label_mapping,
                             file_extension=".jpeg"), LabeledDirDataset(os.path.join(data_folder_path, "test"),
                                                                        label_mapping=label_mapping,
                                                                        file_extension=".jpeg")
