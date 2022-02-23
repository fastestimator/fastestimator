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
"""Siamese Network implementation using Omniglot dataset"""
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

import fastestimator as fe
from fastestimator.backend import feed_forward
from fastestimator.dataset.data import omniglot
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import ShiftScaleRotate
from fastestimator.op.numpyop.univariate import Minmax, ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.adapt import EarlyStopping, LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import Data


def lr_schedule(epoch):
    """Learning rate schedule"""
    lr = 0.0001 * np.power(0.99, epoch)
    return lr


class OneShotAccuracy(Trace):
    """Trace for calculating one shot accuracy"""
    def __init__(self, dataset, model, N=20, trials=400, mode=("eval", "test"), output_name="one_shot_accuracy"):

        super().__init__(mode=mode, outputs=output_name)
        self.dataset = dataset
        self.model = model
        self.total = 0
        self.correct = 0
        self.output_name = output_name
        self.N = N
        self.trials = trials

    def on_epoch_begin(self, data: Data):
        self.total = 0
        self.correct = 0

    def on_epoch_end(self, data: Data):
        for _ in range(self.trials):
            img_path = self.dataset.one_shot_trial(self.N)
            input_img = (np.array([np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1) / 255. for i in img_path[0]],
                                  dtype=np.float32),
                         np.array([np.expand_dims(cv2.imread(i, cv2.IMREAD_GRAYSCALE), -1) / 255. for i in img_path[1]],
                                  dtype=np.float32))
            prediction_score = feed_forward(self.model, input_img, training=False).numpy()

            if np.argmax(prediction_score) == 0 and prediction_score.std() > 0.01:
                self.correct += 1

            self.total += 1

        data.write_with_log(self.outputs[0], self.correct / self.total)


def siamese_network(input_shape=(105, 105, 1), classes=1):
    """Network Architecture"""
    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    # Creating the convnet which shares weights between the left and right legs of Siamese network
    siamese_convnet = Sequential()

    siamese_convnet.add(
        layers.Conv2D(filters=64,
                      kernel_size=10,
                      strides=1,
                      input_shape=input_shape,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=128,
                      kernel_size=7,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=128,
                      kernel_size=4,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=256,
                      kernel_size=4,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.Flatten())

    siamese_convnet.add(
        layers.Dense(4096,
                     activation='sigmoid',
                     kernel_initializer=RandomNormal(mean=0, stddev=0.2),
                     kernel_regularizer=l2(1e-4),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    encoded_left_input = siamese_convnet(left_input)
    encoded_right_input = siamese_convnet(right_input)

    l1_encoded = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_left_input, encoded_right_input])

    output = layers.Dense(classes,
                          activation='sigmoid',
                          kernel_initializer=RandomNormal(mean=0, stddev=0.2),
                          bias_initializer=RandomNormal(mean=0.5, stddev=0.01))(l1_encoded)

    return Model(inputs=[left_input, right_input], outputs=output)


def get_estimator(epochs=200,
                  batch_size=128,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir=None):
    # step 1. prepare pipeline
    train_data, eval_data = omniglot.load_data(root_dir=data_dir)
    test_data = eval_data.split(0.5)

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="x_a", outputs="x_a", color_flag='gray'),
            ReadImage(inputs="x_b", outputs="x_b", color_flag='gray'),
            Sometimes(
                ShiftScaleRotate(image_in="x_a",
                                 image_out="x_a",
                                 shift_limit=0.05,
                                 scale_limit=0.2,
                                 rotate_limit=10,
                                 mode="train"),
                prob=0.89),
            Sometimes(
                ShiftScaleRotate(image_in="x_b",
                                 image_out="x_b",
                                 shift_limit=0.05,
                                 scale_limit=0.2,
                                 rotate_limit=10,
                                 mode="train"),
                prob=0.89),
            Minmax(inputs="x_a", outputs="x_a"),
            Minmax(inputs="x_b", outputs="x_b")
        ])

    # step 2. prepare model
    model = fe.build(model_fn=siamese_network, model_name="siamese_net", optimizer_fn="adam")

    network = fe.Network(ops=[
        ModelOp(inputs=["x_a", "x_b"], model=model, outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss", form="binary"),
        UpdateOp(model=model, loss_name="loss")
    ])

    # step 3.prepare estimator
    traces = [
        LRScheduler(model=model, lr_fn=lr_schedule),
        Accuracy(true_key="y", pred_key="y_pred"),
        OneShotAccuracy(dataset=eval_data, model=model, output_name='one_shot_accuracy'),
        BestModelSaver(model=model, save_dir=save_dir, metric="one_shot_accuracy", save_best_mode="max"),
        EarlyStopping(monitor="one_shot_accuracy", patience=20, compare='max', mode="eval")
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
