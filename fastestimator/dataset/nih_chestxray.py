"""Download NIH-Chest Xray dataset"""
import os
import tarfile
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _download_data(link, data_path, idx, total_idx):
    if not os.path.exists(data_path):
        print("Downloading data to {}, file: {} / {}".format(data_path, idx + 1, total_idx))
        wget.download(link, data_path, bar=bar_custom)


def load_data(path=None):
    """Download the NIH dataset to local storage.

    Args:
        path (str, optional): The path to store the  data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (csv_path, path) tuple, where
        
        * **csv_path** (str) -- Path to the summary csv file containing the following column:
        
            * x (str): Image directory relative to the returned path.
            
        * **path** (str) -- Data folder path.

    """
    if path is None:
        path = os.path.join(str(Path.home()), 'fastestimator_data', 'NIH_Chestxray')
    else:
        path = os.path.join(os.path.abspath(path), 'NIH_Chestxray')
    os.makedirs(path, exist_ok=True)
    #download data
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    data_paths = [os.path.join(path, "images_{}.tar.gz".format(x)) for x in range(len(links))]
    for idx, (link, data_path) in enumerate(zip(links, data_paths)):
        _download_data(link, data_path, idx, len(links))
    #extract data
    image_extracted_path = os.path.join(path, 'images')
    if not os.path.exists(image_extracted_path):
        for idx, data_path in enumerate(data_paths):
            print("Extracting {}, file {} / {}".format(data_path, idx + 1, len(links)))
            with tarfile.open(data_path) as img_tar:
                img_tar.extractall(path)
    #generate_csv
    csv_path = os.path.join(path, 'nih_chestxray.csv')
    if not os.path.exists(csv_path):
        image_names = [
            os.path.relpath(os.path.join(image_extracted_path, x), path) for x in os.listdir(image_extracted_path)
        ]
        df = pd.DataFrame(image_names, columns=["x"])
        df.to_csv(csv_path, index=False)
    return csv_path, path
