import tempfile

import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.dataset.data import imdb_review
from fastestimator.op.numpyop.univariate.reshape import Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

MAX_WORDS = 10000
MAX_LEN = 500


def create_lstm(max_len):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(MAX_WORDS, 64, input_length=max_len))
    model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.CuDNNLSTM(64))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def get_estimator(epochs=10, batch_size=64, steps_per_epoch=None, model_dir=tempfile.mkdtemp()):

    # step 1. prepare data
    train_data, eval_data = imdb_review.load_data(MAX_LEN, MAX_WORDS)

    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=Reshape(1, inputs="y", outputs="y"))

    # step 2. prepare model
    model = fe.build(model_fn=lambda: create_lstm(MAX_LEN), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])

    traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=model_dir)]
    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
