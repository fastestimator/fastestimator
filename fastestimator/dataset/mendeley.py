
import os
import tarfile
import tempfile
from operator import add
import zipfile
import h5py
import numpy as np
import pandas as pd
import wget
import glob
import tensorflow as tf


def img_data_constructor(data_folder, mode, csv_path):

    normal_cases_dir = os.path.join(data_folder, 'NORMAL')
    pneumonia_cases_dir = os.path.join(data_folder, 'PNEUMONIA')
    
    data = []

    for dirpath,_,filenames in os.walk(normal_cases_dir):
       for f in filenames:
           # Select only jpeg files:
           if f[-4:]=='jpeg' :
               img = os.path.abspath(os.path.join(dirpath, f))
               data.append((img,0)) 

    for dirpath,_,filenames in os.walk(pneumonia_cases_dir):
       for f in filenames:
           if f[-4:]=='jpeg':
               img = os.path.abspath(os.path.join(dirpath, f))
               data.append((img,1)) 

    num_example=len(data)
    print("found %d number of examples for %s" % (num_example, mode))

    data = pd.DataFrame(data, columns=['x', 'y'],index=None)
    data = data.sample(frac=1.).reset_index(drop=True)
    data.to_csv(csv_path, index=False)

    return data


def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "Mendeley")
    if not os.path.exists(path):
        os.mkdir(path)

    train_csv = os.path.join(path, "train_data.csv")
    test_csv = os.path.join(path, "test_data.csv")

    if not (os.path.exists(os.path.join(path, "ChestXRay2017.zip")) ) :
        print("downloading data to %s" % path)
        wget.download('https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/ChestXRay2017.zip', path)

    if not (os.path.exists(os.path.join(path, "chest_xray/train")) and os.path.exists(os.path.join(path, "chest_xray/test")) ):
        print(" ")
        print("extracting data...")
        with zipfile.ZipFile(os.path.join(path, "ChestXRay2017.zip"), 'r') as zip:
            zip.extractall(path=path)

    if not (os.path.exists(train_csv) and os.path.exists(test_csv) ):
        print("constructing data for FastEstimator...")
        
        train_folder = os.path.join(path, "chest_xray/train")
        test_folder = os.path.join(path, "chest_xray/test")

        img_data_constructor(train_folder, "train", train_csv)
        img_data_constructor(test_folder, "test", test_csv)

    return train_csv, test_csv, path
