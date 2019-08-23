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
"""Download Movielens-20M dataset.
http://files.grouplens.org/datasets/movielens/ml-20m.zip
"""

import os
import tempfile
import zipfile

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import wget

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX",
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def _preprocess_data(path, data_path):
    movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))

    df = ratings.merge(movies, on="movieId")

    for i in GENRES:
        df[i] = df["genres"].apply(lambda x: 1 if i in x else 0)

    df["genres_count"] = df["genres"].apply(lambda x: len(x.split("|")))
    df["rating"] = df["rating"].apply(lambda x: 1 if x > 3 else 0)

    df = df.drop(columns=["title", "genres"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    df = df.drop(columns=["timestamp"])

    LE_user = LabelEncoder()
    df["user_id"] = LE_user.fit_transform(df["userId"])

    LE_movie = LabelEncoder()
    df["movie_id"] = LE_movie.fit_transform(df["movieId"])

    # Creation of Train and Evaluation Sets
    train_df = df.sample(frac=0.8, random_state=0)
    eval_df = df.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    # Applying Standard Scaler
    Std_cols = ["genres_count", "year", "month", "day"]
    scaler = StandardScaler()
    train_df[Std_cols] = scaler.fit_transform(train_df[Std_cols])
    eval_df[Std_cols] = scaler.transform(eval_df[Std_cols])

    return train_df, eval_df


def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), 'FE_MOVIELENS')
    if not os.path.exists(path):
        os.mkdir(path)
        print("Downloading data to Path: ", path)

    train_data_dir = os.path.join(path, 'ml-20m')
    train_csv_path = os.path.join(path, 'movielens_20m.csv')

    if not os.path.exists(train_data_dir) or not os.listdir(train_data_dir):
        if not os.path.exists(os.path.join(path, 'ml-20m.zip')):
            wget.download(
                'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
                path)
        print(f'extracting data in {path} ....')
        zippath = os.path.join(path, 'ml-20m.zip')
        zipfile.ZipFile(zippath).extractall(path)

    print("Processing Dataset")
    train_csv, eval_csv = _preprocess_data(path, train_data_dir)

    #restructuring datasets
    x_train = train_csv[["user_id", "movie_id"] + GENRES +
                        ["genres_count", "year", "month", "day"]]
    y_train = train_csv[["rating"]]

    x_eval = eval_csv[["user_id", "movie_id"] + GENRES +
                      ["genres_count", "year", "month", "day"]]
    y_eval = eval_csv[["rating"]]

    return (x_train, y_train), (x_eval, y_eval)
