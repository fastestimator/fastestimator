import os
import tempfile
import zipfile
import wget

import pandas as pd
from glob import glob

NUM_IMG_FILES = 82783


def _download_and_extract(url, filename, path):
    if not os.path.exists(os.path.join(path, filename)):
        print("Downloading data to {}".format(path))
        wget.download(url, path)

    with zipfile.ZipFile(os.path.join(path, filename), 'r') as zip_file:
        print("Extracting {}".format(os.path.join(path, filename)))
        zip_file.extractall(path)


def _create_csv(data_path):
    filenames = glob(os.path.join(data_path, '*.jpg'))
    csv = None
    if len(filenames) == NUM_IMG_FILES:
        csv = pd.DataFrame()
        rel_filenames = [f.replace(data_path, '.') for f in filenames]
        csv["image"] = rel_filenames
    else:
        print("One or more images are missing")
    return csv


def load_data(path=None):
    url = {"train": "http://images.cocodataset.org/zips/train2014.zip"}
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "MSCOCO2014")
    path = os.path.abspath(path)

    os.makedirs(path, exist_ok=True)
    train_csv_path = os.path.join(path, "coco_train.csv")
    train_data_dir = os.path.join(path, "train2014")
    if not (os.path.exists(train_data_dir)):
        _download_and_extract(url["train"], "train2014.zip", path)
        train_csv = _create_csv(train_data_dir)
        train_csv.to_csv(train_csv_path, index=False)
    else:
        if not (len(glob(os.path.join(train_data_dir, '*.jpg'))) == NUM_IMG_FILES):
            print("One or more images are missing")
            _download_and_extract(url["train"], "train2014.zip", path)
            train_csv = _create_csv(train_data_dir)
            train_csv.to_csv(train_csv_path, index=False)
        else:
            if not (os.path.exists(train_csv_path)):
                train_csv = _create_csv(train_data_dir)
                train_csv.to_csv(train_csv_path, index=False)
            else:
                print("reusing existing dataset")
    return train_csv_path, train_data_dir
