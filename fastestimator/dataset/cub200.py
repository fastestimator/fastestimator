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
    if path:
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(tempfile.gettempdir(), 'FE_CUB200')

    url = {'image': 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz',
           'annotation': 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz'}

    img_path = os.path.join(path, 'images.tgz')
    anno_path = os.path.join(path, 'annotations.tgz')

    if not (os.path.exists(img_path) and os.path.exists(anno_path)):
        print("Downloading data to {}:".format(path))
        wget.download(url['image'], path)
        wget.download(url['annotation'], path)

    print('\nExtracting files...')
    with tarfile.open(img_path) as img_tar:
        img_tar.extractall(path)
    with tarfile.open(anno_path) as anno_tar:
        anno_tar.extractall(path)

    img_list = glob(os.path.join(path, 'images', '**', '*.jpg'))

    df = pd.DataFrame(data={'image': img_list})
    df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')

    if not df['annotation'].apply(os.path.exists).all():
        raise FileNotFoundError

    df.to_csv(os.path.join(path, 'cub200.csv'), index=False)
    print('Data summary is saved at {}'.format(os.path.join(path, 'cub200.csv')))
