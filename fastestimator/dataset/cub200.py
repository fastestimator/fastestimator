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
"""Download Caltech-UCSD Birds 200 dataset.
http://www.vision.caltech.edu/visipedia/CUB-200.html
"""
import os
import tarfile
import tempfile
from glob import glob

import pandas as pd
import wget


def load_data(path=None):
    """Download the CUB200 data set to local storage. This will generate a cub200.csv file, which contains all the path
        information.

    Args:
        path (str, optional): The path to store the CUB200 data. Defaults to None, will save at `tempfile.gettempdir()`.

    Raises:
        FileNotFoundError: When the gernerated CSV file does not match with the extracted dataset.
    """
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "CUB200")
    if not os.path.exists(path):
        os.makedirs(path)
    csv_path = os.path.join(path, 'cub200.csv')
    image_compressed_path = os.path.join(path, 'images.tgz')
    annotation_compressed_path = os.path.join(path, 'annotations.tgz')
    image_extracted_path = os.path.join(path, 'images')
    annotation_extracted_path = os.path.join(path, 'annotations-mat')
    if not (os.path.exists(image_compressed_path) and os.path.exists(annotation_compressed_path)):
        print("Downloading data to {} ...".format(path))
        wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz', path)
        wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz', path)
    if not (os.path.exists(image_extracted_path) and os.path.exists(annotation_extracted_path)):
        print('\nExtracting files ...')
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(path)
        with tarfile.open(annotation_compressed_path) as anno_tar:
            anno_tar.extractall(path)
    if not os.path.exists(csv_path):
        img_list = glob(os.path.join(path, 'images', '**', '*.jpg'))
        df = pd.DataFrame(data={'image': img_list})
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, path))
        df['image'] = df['image'].apply(os.path.normpath)
        df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')
        df.to_csv(csv_path, index=False)
        print('Data summary is saved at {}'.format(csv_path))
    return csv_path, path
