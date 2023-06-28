# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
from pathlib import Path
from typing import Tuple

import numpy as np

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.google_download_util import download_file_from_google_drive

dataset_ids = {
    "chestmnist": "1lGZtFjlviRf_vwmxB4NgApjCdfDuh8pk",
    "adrenalmnist3d": "1E__zT_PqXlaiyW3r6Ii98UVVv5-nDZ_q",
    "bloodmnist": "17r9NENUnUoqnWZ-zJaveLU40R746A-ul",
    "breastmnist": "1E3fWybVlOXSIjg_kFSQ-nwOikLgrbsJc",
    "dermamnist": "1k-vw5otV5rCQLVQKyxmo1mhI3KqDbwU5",
    "fracturemnist3d": "104OYiONKAsN2O-VeVGS7ReXZI4GtNsZk",
    "nodulemnist3d": "1krf_Uv-bA5CEt_dCD6h8xJbzZk9ETWLU",
    "octmnist": "1yRhEBDQwHu5jSnrzIm87Cazns3Eesv4b",
    "organamnist": "1yRhEBDQwHu5jSnrzIm87Cazns3Eesv4b",
    "organcmnist": "1GqXyaV6arJq0O-3Fn2hvBw77MG9fHeoz",
    "organmnist3d": "1RQiHLj35u3m6GCiBKJFnb9VH76u7yRi0",
    "organsmnist": "1spWIVxKaLvWAxLHGLSr0IJJBSm0MDBB4",
    "pathmnist": "12He-AovBA5ReIg1snhA6Jm2jQCvKKYp7",
    "pneumoniamnist": "1ObI6UzRYZby9YCmk53m1WCXo747wzNqT",
    "retinamnist": "1dugDJx_Z9nvtbD9SidxkVjgvysA93IG7",
    "synapsemnist3d": "1JCBppD-bCYQhjwufJ63OS8XgMaT3sRv5",
    "tissuemnist": "1M5wec6b-iiLPjVs3MrCrAlKCxwjuEL2z",
    "vesselmnist3d": "1orrps7d-SKB4kFPV0jaN3r_jfYpDtGpP",
}


def load_data(
    dataset_name: str,
    root_dir: str = None,
    image_key: str = "x",
    label_key: str = "y",
) -> Tuple[NumpyDataset, NumpyDataset, NumpyDataset]:
    """
    Download and load the medmnist data. Medmnist has 18 datasets. Here is the list of datasets available in Medmnist.
    [
        'chestmnist',
        'adrenalmnist3d',
        'bloodmnist',
        'breastmnist',
        'dermamnist',
        'fracturemnist3d',
        'nodulemnist3d',
        'octmnist',
        'organamnist',
        'organcmnist',
        'organmnist3d',
        'organsmnist',
        'pathmnist',
        'pneumoniamnist',
        'retinamnist',
        'synapsemnist3d',
        'tissuemnist',
        'vesselmnist3d'
    ]
    For more details on the dataset, please check https://medmnist.com, https://zenodo.org/record/6496656

    Args:
        dataset_name (str): Name of the dataset to download and load.
        root_dir (str, optional): The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory. Defaults to None.
        image_key (str, optional): The key for image. Defaults to "x".
        label_key (str, optional): The key for label. Defaults to "y".

    Raises:
        ValueError: if the dataset_name is invalid

    Returns:
        Tuple[NumpyDataset, NumpyDataset, NumpyDataset]: returns a tuple of traing, val and test data.
    """
    if dataset_name not in dataset_ids:
        raise ValueError("Invalid value for dataset_name.")

    if root_dir is None:
        home = str(Path.home())
        root_dir = os.path.join(home, "fastestimator_data", "medmnist")
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), "medmnist")

    os.makedirs(root_dir, exist_ok=True)

    download_path = os.path.join(root_dir, f"{dataset_name}.npz")

    print("Downloading data to {}".format(root_dir))
    download_file_from_google_drive(dataset_ids[dataset_name], download_path)

    npz_file = np.load(download_path)

    x_train = npz_file["train_images"]
    y_train = npz_file["train_labels"]

    x_val = npz_file["val_images"]
    y_val = npz_file["val_labels"]

    x_test = npz_file["test_images"]
    y_test = npz_file["test_labels"]

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    val_data = NumpyDataset({image_key: x_val, label_key: y_val})
    test_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, val_data, test_data
