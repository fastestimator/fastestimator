# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import shutil
from pathlib import Path
from typing import Tuple, TypeVar

import numpy as np
import requests
import wget
from tqdm import tqdm

from fastestimator.dataset.data.cifair10 import _load_batch
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.wget_util import callback_progress

wget.callback_progress = callback_progress
Response = TypeVar('Response', bound=requests.models.Response)


def _get_confirm_token(response: Response) -> str:
    """Retrieve the token from the cookie jar of HTTP request to keep the session alive.
    Args:
        response: Response object of the HTTP request.
    Returns:
        The value of cookie in the response object.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _download_file_from_google_drive(file_id: str, destination: str) -> None:
    """Download the data from the Google drive public URL.

    This method will create a session instance to persist the requests and reuse TCP connection for the large files.

    Args:
        file_id: File ID of Google drive URL.
        destination: Destination path where the data needs to be stored.
    """
    URL = "https://drive.google.com/uc?export=download&confirm=t"
    CHUNK_SIZE = 128
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    total_size = int(response.headers.get('Content-Length', 0))
    progress = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()


def load_data(root_dir: str = None, image_key: str = "x", label_key: str = "y",
              label_mode: str = "fine") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the ciFAIR100 dataset.

    This is the cifar100 dataset but with test set duplicates removed and replaced. See
    https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
    dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, test_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'ciFAIR100')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'ciFAIR100')
    os.makedirs(root_dir, exist_ok=True)

    image_compressed_path = os.path.join(root_dir, 'ciFAIR100.zip')
    image_extracted_path = os.path.join(root_dir, 'ciFAIR-100')

    if not os.path.exists(image_extracted_path):
        if not os.path.exists(image_compressed_path):
            print("Downloading data to {}".format(root_dir))
            _download_file_from_google_drive('1ZE_wf5UTd9fJqBgikb7MJtfFEeAfXybS', image_compressed_path)

        print("Extracting data to {}".format(root_dir))
        shutil.unpack_archive(image_compressed_path, root_dir)

    fpath = os.path.join(image_extracted_path, 'train')
    x_train, y_train = _load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(image_extracted_path, 'test')
    x_test, y_test = _load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    test_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, test_data
