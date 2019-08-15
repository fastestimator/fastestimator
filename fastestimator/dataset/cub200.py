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
    """Download the CUB200 data set to local storage. This will generate a cub200.csv file, which contains all the path information.

    Args:
        path (str, optional): The path to store the CUB200 data. Defaults to None, will save at `tempfile.gettempdir()`.

    Raises:
        FileNotFoundError: When the gernerated CSV file does not match with the extracted dataset.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(tempfile.gettempdir(), 'FE_CUB200')

    csv_path = os.path.join(path, 'cub200.csv')

    if os.path.isfile(csv_path):
        print('Found existing {}.'.format(csv_path))
        df = pd.read_csv(csv_path)
        found_images = df['image'].apply(lambda x: os.path.join(path, x)).apply(os.path.isfile).all()
        found_annoation = df['annotation'].apply(lambda x: os.path.join(path, x)).apply(os.path.isfile).all()
        if not (found_images and found_annoation):
            print('There are missing files. Will download dataset again.')
        else:
            print('All files exist, using existing {}.'.format(csv_path))
            return csv_path, path

    url = {
        'image': 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz',
        'annotation': 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz'
    }

    img_path = os.path.join(path, 'images.tgz')
    anno_path = os.path.join(path, 'annotations.tgz')

    print("Downloading data to {} ...".format(path))
    wget.download(url['image'], path)
    wget.download(url['annotation'], path)

    print('\nExtracting files ...')
    with tarfile.open(img_path) as img_tar:
        img_tar.extractall(path)
    with tarfile.open(anno_path) as anno_tar:
        anno_tar.extractall(path)

    img_list = glob(os.path.join(path, 'images', '**', '*.jpg'))

    df = pd.DataFrame(data={'image': img_list})
    df['image'] = df['image'].apply(lambda x: os.path.relpath(x, path))
    df['image'] = df['image'].apply(os.path.normpath)
    df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')

    if not (df['annotation'].apply(lambda x: os.path.join(path, x))).apply(os.path.exists).all():
        raise FileNotFoundError

    df.to_csv(os.path.join(path, 'cub200.csv'), index=False)
    print('Data summary is saved at {}'.format(csv_path))

    return csv_path, path
