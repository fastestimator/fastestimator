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
import os

import fastestimator as fe
import tensorflow as tf
import pandas as pd
import numpy as np

from fastestimator.op.tensorop.loss import Loss
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop import Resize, Minmax
from fastestimator.op.numpyop import ImageReader
from fastestimator.layers import GradReversal
from fastestimator.dataset import mnist, usps
from fastestimator.trace import Trace

from tensorflow.keras import layers, Model, losses, backend


class GRLWeightController(Trace):
    def __init__(self, alpha):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.alpha = alpha

    def on_begin(self, state):
        self.total_steps = state['total_train_steps']

    def on_batch_begin(self, state):
        p = state['train_step'] / self.total_steps
        current_alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        backend.set_value(self.alpha, current_alpha)


class FELoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.label_loss_obj = losses.SparseCategoricalCrossentropy(reduction=losses.Reduction.NONE)
        self.domain_loss_obj = losses.BinaryCrossentropy(reduction=losses.Reduction.NONE)

    def forward(self, data, state):
        src_c_logit, src_c_label, src_d_logit, tgt_d_logit = data
        c_loss = self.label_loss_obj(y_true=src_c_label, y_pred=src_c_logit)
        src_d_loss = self.domain_loss_obj(y_true=tf.zeros_like(src_d_logit), y_pred=src_d_logit)
        tgt_d_loss = self.domain_loss_obj(y_true=tf.ones_like(tgt_d_logit), y_pred=tgt_d_logit)
        return c_loss + src_d_loss + tgt_d_loss


def build_feature_extractor(img_shape=(28, 28, 1)):
    x0 = layers.Input(shape=img_shape)
    x = layers.Conv2D(32, 5, activation="relu", padding="same")(x0)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(48, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    feat_map = layers.Flatten()(x)
    return Model(inputs=x0, outputs=feat_map)


def build_label_predictor(feat_dim):
    x0 = layers.Input(shape=(feat_dim, ))
    x = layers.Dense(100, activation="relu")(x0)
    x = layers.Dense(100, activation="relu")(x)
    return Model(inputs=x0, outputs=x)


def build_domain_predictor(feat_dim, alpha):
    x0 = layers.Input(shape=(feat_dim, ))
    x = GradReversal(l=alpha)(x0)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs=x0, outputs=x)


def get_estimator(batch_size=128, epochs=100):

    usps_train_csv, _, usps_parent_dir = usps.load_data()
    mnist_train_csv, _, mnist_parent_dir = mnist.load_data()

    df = pd.read_csv(mnist_train_csv)
    df.columns = ['source_img', 'source_label']
    df.to_csv(mnist_train_csv, index=False)

    df = pd.read_csv(usps_train_csv)
    df.columns = ['target_img', 'target_label']
    df.to_csv(usps_train_csv, index=False)



    writer = fe.RecordWriter(save_dir=os.path.join(os.path.dirname(mnist_parent_dir), 'dann', 'tfr'),
                        train_data=(usps_train_csv, mnist_train_csv),
                        ops=(
                            [ImageReader(inputs="target_img", outputs="target_img", parent_path=usps_parent_dir, grey_scale=True)], # first tuple element
                            [ImageReader(inputs="source_img", outputs="source_img", parent_path=mnist_parent_dir, grey_scale=True)])) # second tuple element

    pipeline = fe.Pipeline(
        batch_size=batch_size,
        data=writer,
        ops=[
            Resize(inputs="target_img", outputs="target_img", size=(28, 28)),
            Resize(inputs="source_img", outputs="source_img", size=(28, 28)),
            Minmax(inputs="target_img", outputs="target_img"),
            Minmax(inputs="source_img", outputs="source_img")
        ])

    alpha = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    img_shape = (28, 28, 1)
    feat_dim = 7 * 7 * 48

    feature_extractor = fe.build(model_def=lambda: build_feature_extractor(img_shape),
                                 model_name="feature_extractor",
                                 loss_name="fe_loss",
                                 optimizer=tf.keras.optimizers.Adam(1e-4))

    label_predictor = fe.build(model_def=lambda: build_label_predictor(feat_dim),
                               model_name="label_predictor",
                               loss_name="fe_loss",
                               optimizer=tf.keras.optimizers.Adam(1e-4))

    domain_predictor = fe.build(model_def=lambda: build_domain_predictor(feat_dim, alpha),
                                model_name="domain_predictor",
                                loss_name="fe_loss",
                                optimizer=tf.keras.optimizers.Adam(1e-4))

    network = fe.Network(ops=[
        ModelOp(inputs="source_img", outputs="src_feat", model=feature_extractor),
        ModelOp(inputs="target_img", outputs="tgt_feat", model=feature_extractor),
        ModelOp(inputs="src_feat", outputs="src_c_logit", model=label_predictor),
        ModelOp(inputs="src_feat", outputs="src_d_logit", model=domain_predictor),
        ModelOp(inputs="tgt_feat", outputs="tgt_d_logit", model=domain_predictor),
        FELoss(inputs=("src_c_logit", "source_label", "src_d_logit", "tgt_d_logit"), outputs="fe_loss")
    ])

    traces = [GRLWeightController(alpha=alpha)]

    estimator = fe.Estimator(pipeline=pipeline, network=network, traces=traces, epochs=epochs)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
