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
import random
import tarfile
from pathlib import Path
from typing import Optional, TypeVar

import pandas as pd
import requests
from tqdm import tqdm

from fastestimator.dataset.csv_dataset import CSVDataset

Response = TypeVar('Response', bound=requests.models.Response)


def _download_file_from_google_drive(id: str, destination: str) -> None:
    """Download the data from the Google drive public URL.

    This method will create a session instance to persist the requests and reuse TCP connection for the large files.

    Args:
        id: File ID of Google drive URL.
        destination: Destination path where the data needs to be stored.
    """
    URL = "https://drive.google.com/uc?export=download"
    CHUNK_SIZE = 128
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()


def _get_confirm_token(response: Response) -> str:
    """Retrieve the token from the cookie jar of HTTP request to keep the session alive.

    Args:
        response: Response object of the HTTP request.

    Returns"
        The value of cookie in the response object.

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def load_data(root_dir: Optional[str] = None) -> CSVDataset:
    """Load and return the Caltech-UCSD Birds 200 (CUB200) dataset.

    Sourced from http://www.vision.caltech.edu/visipedia/CUB-200.html. This method will download the data to local
        storage if the data has not been previously downloaded.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        train_data
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'CUB200')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'CUB200')
    os.makedirs(root_dir, exist_ok=True)

    csv_path = os.path.join(root_dir, 'cub200.csv')
    image_compressed_path = os.path.join(root_dir, 'images.tgz')
    annotation_compressed_path = os.path.join(root_dir, 'annotations.tgz')
    image_extracted_path = os.path.join(root_dir, 'images')
    annotation_extracted_path = os.path.join(root_dir, 'annotations-mat')

    if not (os.path.exists(image_extracted_path) and os.path.exists(annotation_extracted_path)):
        # download
        if not (os.path.exists(image_compressed_path) and os.path.exists(annotation_compressed_path)):
            print("Downloading data to {}".format(root_dir))
            _download_file_from_google_drive('1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx', image_compressed_path)
            _download_file_from_google_drive('16NsbTpMs5L6hT4hUJAmpW2u7wH326WTR', annotation_compressed_path)

        # extract
        print("\nExtracting files ...")
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(root_dir)
        with tarfile.open(annotation_compressed_path) as anno_tar:
            anno_tar.extractall(root_dir)

    # glob and generate csv
    if not os.path.exists(csv_path):
        image_folder = os.path.join(root_dir, "images")
        class_names = os.listdir(image_folder)
        label_map = {}
        images = []
        labels = []
        idx = 0
        for class_name in class_names:
            if not class_name.startswith("._"):
                image_folder_class = os.path.join(image_folder, class_name)
                label_map[class_name] = idx
                idx += 1
                image_names = os.listdir(image_folder_class)
                for image_name in image_names:
                    if not image_name.startswith("._"):
                        images.append(os.path.join(image_folder_class, image_name))
                        labels.append(label_map[class_name])
        zipped_list = list(zip(images, labels))
        random.shuffle(zipped_list)
        df = pd.DataFrame(zipped_list, columns=["image", "label"])
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, root_dir))
        df['image'] = df['image'].apply(os.path.normpath)
        df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')
        df.to_csv(csv_path, index=False)
        print("Data summary is saved at {}".format(csv_path))
    return CSVDataset(csv_path)
