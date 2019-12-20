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
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import fastestimator as fe
from fastestimator.dataset import mnist, usps
from fastestimator.op.numpyop import ImageReader
from fastestimator import RecordWriter
from fastestimator.op.tensorop import Resize, Minmax
from tensorflow.keras import layers, Model, Sequential
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Loss, ModelOp
from fastestimator.trace import Trace


class ExtractSourceFeature(TensorOp):
    def __init__(self, model_path, inputs, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.source_feature_extractor = tf.keras.models.load_model(model_path, compile=False)
        self.source_feature_extractor.trainable = False

    def forward(self, data, state):
        return self.source_feature_extractor(data)


class FELoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, data, state):
        target_score = data
        return self.cross_entropy(tf.ones_like(target_score), target_score)


class DLoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, data, state):
        source_score, target_score = data
        source_loss = self.cross_entropy(tf.ones_like(source_score), source_score)
        target_loss = self.cross_entropy(tf.zeros_like(target_score), target_score)
        total_loss = source_loss + target_loss
        return 0.5 * total_loss


class EvaluateTargetClassifier(Trace):
    def __init__(self, model_name, model_path):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.target_model = tf.keras.Sequential()
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def on_begin(self, state):
        self.target_model.add(self.network.model[self.model_name])
        self.target_model.add(tf.keras.models.load_model(self.model_path))

        def on_batch_end(self, state):
            if state["epoch"] == 0 or state["epoch"] == 99:
                target_img, target_label = state["batch"]["target_img"], state["batch"]["target_label"]
                logits = self.target_model(target_img)
                self.acc_metric(target_label, logits)

    def on_epoch_end(self, state):
        if state["epoch"] == 0 or state["epoch"] == 99:
            acc = self.acc_metric.result()
            print("FastEstimator-EvaluateTargetClassifier: %0.4f at epoch %d" % (acc, state["epoch"]))
            self.acc_metric.reset_states()


class LoadPretrainedFE(Trace):
    def __init__(self, model_name, model_path):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path

    def on_begin(self, state):
        self.network.model[self.model_name].load_weights(self.model_path)
        print("FastEstimator-LoadPretrainedFE: loaded pretrained feature extractor")


def build_feature_extractor(input_shape=(32, 32, 1), feature_dim=512):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(feature_dim, activation='relu'))
    return model


def build_classifer(feature_dim=512, num_classes=10):
    model = Sequential()
    model.add(layers.Dense(num_classes, activation='softmax', input_dim=feature_dim))
    return model


def build_discriminator(feature_dim=512):
    model = Sequential()
    model.add(layers.Dense(1024, activation='relu', input_dim=feature_dim))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def get_estimator(pretrained_fe_path, classifier_path, data_path=None, epochs=100):

    assert os.path.exists(pretrained_fe_path), "Pretrained feature extractor is missing"
    assert os.path.exists(classifier_path), "Pretrained classifier is missing"
    usps_train_csv, usps_eval_csv, usps_parent_dir = usps.load_data(data_path)
    mnist_train_csv, mnist_eval_csv, mnist_parent_dir = mnist.load_data(data_path)

    tfr_path = os.path.join(os.path.dirname(usps_parent_dir), 'ADDA-tfrecords')
    os.makedirs(tfr_path, exist_ok=True)

    df = pd.read_csv(usps_train_csv)
    df.columns = ['target_img', 'target_label']
    df.to_csv(usps_train_csv, index=False)

    df = pd.read_csv(usps_eval_csv)
    df.columns = ['target_img', 'target_label']
    df.to_csv(usps_eval_csv, index=False)

    df = pd.read_csv(mnist_train_csv)
    df.columns = ['source_img', 'source_label']
    df.to_csv(mnist_train_csv, index=False)

    df = pd.read_csv(mnist_eval_csv)
    df.columns = ['source_img', 'source_label']
    df.to_csv(mnist_eval_csv, index=False)

    BATCH_SIZE = 128

    writer = RecordWriter(save_dir=tfr_path,
                          train_data=(usps_train_csv, mnist_train_csv),
                          ops=(
                              [ImageReader(inputs="target_img", outputs="target_img", parent_path=usps_parent_dir, grey_scale=True)], # first tuple element
                              [ImageReader(inputs="source_img", outputs="source_img", parent_path=mnist_parent_dir, grey_scale=True)])) # second tuple element

    pipeline = fe.Pipeline(
        batch_size=BATCH_SIZE,
        data=writer,
        ops=[
            Resize(inputs="target_img", outputs="target_img", size=(32, 32)),
            Resize(inputs="source_img", outputs="source_img", size=(32, 32)),
            Minmax(inputs="target_img", outputs="target_img"),
            Minmax(inputs="source_img", outputs="source_img")
        ])

    # Step2: Define Network
    feature_extractor = fe.build(model_def=build_feature_extractor,
                                 model_name="fe",
                                 loss_name="fe_loss",
                                 optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9))

    discriminator = fe.build(model_def=build_discriminator,
                             model_name="disc",
                             loss_name="d_loss",
                             optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9))

    network = fe.Network(ops=[
        ModelOp(inputs="target_img", outputs="target_feature", model=feature_extractor),
        ModelOp(inputs="target_feature", outputs="target_score", model=discriminator),
        ExtractSourceFeature(model_path=pretrained_fe_path, inputs="source_img", outputs="source_feature"),
        ModelOp(inputs="source_feature", outputs="source_score", model=discriminator),
        DLoss(inputs=("source_score", "target_score"), outputs="d_loss"),
        FELoss(inputs="target_score", outputs="fe_loss")
    ])

    traces = [
        LoadPretrainedFE(model_name="fe", model_path=pretrained_fe_path),
        EvaluateTargetClassifier(model_name="fe", model_path=classifier_path)
    ]

    estimator = fe.Estimator(pipeline=pipeline, network=network, traces=traces, epochs=epochs)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
