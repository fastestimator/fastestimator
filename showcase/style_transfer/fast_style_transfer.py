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
import cv2
import numpy as np
import tensorflow as tf

from fastestimator.architecture.stnet import styleTransferNet, lossNet
from fastestimator.dataset.mscoco import load_data
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.record.preprocess import ImageReader, Resize
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import TensorOp


class Rescale(TensorOp):
    def forward(self, data, state):
        return (tf.cast(data, tf.float32) - 127.5) / 127.5


class ExtractVGGFeatures(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs, outputs, mode)
        self.vgg = lossNet()

    def forward(self, data, state):
        return self.vgg(data)


class StyleContentLoss(Loss):
    def __init__(self, style_weight, content_weight, tv_weight, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

    def calculate_style_recon_loss(self, y_true, y_pred):
        y_true_gram = self.calculate_gram_matrix(y_true)
        y_pred_gram = self.calculate_gram_matrix(y_pred)
        y_diff_gram = y_pred_gram - y_true_gram
        y_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(y_diff_gram), axis=(1, 2)))
        return tf.reduce_mean(y_norm)

    def calculate_feature_recon_loss(self, y_true, y_pred):
        y_diff = y_pred - y_true
        num_elts = tf.cast(tf.reduce_prod(y_diff.shape[1:]), tf.float32)
        y_diff_norm = tf.reduce_sum(tf.square(y_diff), axis=(1, 2, 3)) / num_elts
        return tf.reduce_mean(y_diff_norm)

    def calculate_gram_matrix(self, x):
        x = tf.cast(x, tf.float32)
        num_elts = tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)
        gram_matrix = tf.einsum('bijc,bijd->bcd', x, x)
        gram_matrix /= num_elts
        return gram_matrix

    def calculate_total_variation(self, y_pred):
        return tf.reduce_mean(tf.image.total_variation(y_pred))

    def forward(self, data, state):
        y_pred, y_style, y_content, image_out = data

        style_loss = [self.calculate_style_recon_loss(a, b) for a, b in zip(y_style['style'], y_pred['style'])]
        style_loss = tf.add_n(style_loss)
        style_loss *= self.style_weight

        content_loss = [
            self.calculate_feature_recon_loss(a, b) for a, b in zip(y_content['content'], y_pred['content'])
        ]
        content_loss = tf.add_n(content_loss)
        content_loss *= self.content_weight

        total_variation_reg = self.calculate_total_variation(image_out)
        total_variation_reg *= self.tv_weight
        return style_loss + content_loss + total_variation_reg


def get_estimator(style_img_path, data_path=None, style_weight=5.0, content_weight=1.0, tv_weight=1e-4):
    train_csv, path = load_data(data_path)
    style_img = cv2.imread(style_img_path)
    assert style_img is not None, "Invalid style reference image"
    style_img = (style_img.astype(np.float32) / 127.5) / 127.5
    style_img_t = tf.convert_to_tensor(np.expand_dims(style_img, axis=0))
    writer = RecordWriter(
        train_data=train_csv,
        ops=[
            ImageReader(inputs="image", parent_path=path, outputs="image"),
            Resize(inputs="image", target_size=(256, 256), outputs="image")
        ])
    pipeline = Pipeline(batch_size=4, data=writer, ops=[Rescale(inputs="image", outputs="image")])
    model = build(
        keras_model=styleTransferNet(),
        loss=StyleContentLoss(style_weight,
                              content_weight,
                              tv_weight,
                              inputs=('y_pred', 'y_style', 'y_content', 'image_out')),
        optimizer=tf.keras.optimizers.Adam())
    network = Network(ops=[
        ModelOp(inputs="image", model=model, outputs="image_out"),
        ExtractVGGFeatures(inputs=lambda: style_img_t, outputs="y_style"),
        ExtractVGGFeatures(inputs="image", outputs="y_content"),
        ExtractVGGFeatures(inputs="image_out", outputs="y_pred")
    ])
    estimator = Estimator(network=network, pipeline=pipeline, epochs=2)
    return estimator
