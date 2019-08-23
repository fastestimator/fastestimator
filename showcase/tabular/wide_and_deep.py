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
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Embedding, Dense, Flatten, concatenate
from tensorflow.python.keras.models import Model

from sklearn.preprocessing import StandardScaler

from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.network.network import Network
from fastestimator.network.model import ModelOp, build
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.estimator.trace import Accuracy
from fastestimator.dataset import movielens20m_data

_USER_INPUT_DIM = 138493
_MOVIE_INPUT_DIM = 26744
_USER_EMBEDDING_DIM = 64
_MOVIE_EMBEDDING_DIM = 15

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX",
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def wide_and_deep():
    """wide and deep model.
    
    Returns:
        A `Model` object.
    """

    # Network Creation
    input_user = Input(shape=(1, ))
    input_movie = Input(shape=(1, ))
    input_genres = Input(shape=(len(GENRES) + 1, ))
    input_time = Input(shape=(3, ))

    field_embedding = []

    embed_user = Embedding(
        input_dim=_USER_INPUT_DIM,
        output_dim=_USER_EMBEDDING_DIM,
        input_length=1)(input_user)
    field_embedding.append(embed_user)

    embed_movie = Embedding(
        input_dim=_MOVIE_INPUT_DIM,
        output_dim=_MOVIE_EMBEDDING_DIM,
        input_length=1)(input_movie)
    field_embedding.append(embed_movie)

    embed_layer = concatenate(field_embedding, axis=-1)
    embed_layer = Flatten()(embed_layer)

    # deep Part
    x_deep1 = concatenate([embed_layer, input_genres, input_time])
    x_deep2 = Dense(256, activation="relu")(x_deep1)
    x_deep3 = Dense(128, activation="relu")(x_deep2)
    x_deep4 = Dense(64, activation="relu")(x_deep3)

    # wide part
    x_wide1 = concatenate([input_genres, input_time])
    x_wide2 = Dense(16, activation="relu")(x_wide1)

    # combining wide and deep part
    x1 = concatenate([x_wide2, x_deep4])
    x2 = Dense(16, activation="relu")(x1)
    x_output = Dense(1, activation="sigmoid")(x2)

    model = Model(
        inputs=[input_user, input_movie, input_genres, input_time],
        outputs=x_output)

    return model


def get_estimator(epochs=10, batch_size=1024):
    # load data
    (x_train, y_train), (x_eval, y_eval) = movielens20m_data.load_data()

    data = {
        "train": {
            "x_user":
            np.array(x_train["user_id"], dtype="int32"),
            "x_movie":
            np.array(x_train["movie_id"], dtype="int32"),
            "x_genre":
            np.array(x_train[GENRES + ["genres_count"]], dtype="float32"),
            "x_time":
            np.array(x_train[["year", "month", "day"]], dtype="float32"),
            "y":
            np.array(y_train)
        },
        "eval": {
            "x_user":
            np.array(x_eval["user_id"], dtype="int32"),
            "x_movie":
            np.array(x_eval["movie_id"], dtype="int32"),
            "x_genre":
            np.array(x_eval[GENRES + ["genres_count"]], dtype="float32"),
            "x_time":
            np.array(x_eval[["year", "month", "day"]], dtype="float32"),
            "y":
            np.array(y_eval)
        }
    }

    pipeline = Pipeline(batch_size=batch_size, data=data)

    # prepare model
    model = build(
        keras_model=wide_and_deep(),
        loss=BinaryCrossentropy(y_true="y", y_pred="y_pred"),
        optimizer=tf.optimizers.Adam(learning_rate=0.001))
    network = Network(
        ops=ModelOp(
            inputs=["x_user", "x_movie", "x_genre", "x_time"],
            model=model,
            outputs="y_pred"))

    # prepare estimator
    estimator = Estimator(
        network=network,
        pipeline=pipeline,
        epochs=epochs,
        traces=Accuracy(true_key="y", pred_key="y_pred"))

    return estimator
