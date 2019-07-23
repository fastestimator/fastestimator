"""
Download horse2zebra dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
"""
import os
import zipfile
import tempfile
from glob import glob

import pandas as pd
import wget


def load_data(path=None):
    if path:
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(tempfile.gettempdir(), 'FE_HORSE2ZEBRA')

    url = {'image': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'}
    img_path = os.path.join(path, 'horse2zebra.zip')

    if not (os.path.exists(img_path)):
        print("Downloading data to {}:".format(path))
        wget.download(url['image'], path)

    print('\nExtracting files...')
    with zipfile.ZipFile(img_path, 'r') as img_zip:
        img_zip.extractall(path)

    trainA_img = glob(os.path.join(path, 'horese2zebra', 'trainA', '*.jpg'))
    trainA_label = [0] * len(trainA_img)
    trainB_img = glob(os.path.join(path, 'horse2zebra', 'trainB', '*.jpg'))
    trainB_label = [1] * len(trainB_img)

    train_img = trainA_img + trainB_img
    train_label = trainA_label + trainB_label

    train_df = pd.DataFrame()
    train_df['img'] = train_img
    train_df['label'] = train_label
    train_df.to_csv(os.path.join(path, 'train.csv'), index=False)

    testA_img = glob(os.path.join(path, 'horese2zebra', 'testA', '*.jpg'))
    testA_label = [0] * len(testA_img)
    testB_img = glob(os.path.join(path, 'horese2zebra', 'testB', '*.jpg'))
    testB_label = [1]* len(testB_img)
    test_img = testA_img + testB_img
    test_label = testA_label + testB_label
    
    test_df = pd.DataFrame()
    test_df['img'] = test_img
    test_df['label'] = test_label
    test_df.to_csv(os.path.join(path, 'val.csv'), index=False)
    
