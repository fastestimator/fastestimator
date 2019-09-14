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
import tempfile

import cv2
import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.stnet import lossNet, styleTransferNet
from fastestimator.dataset.mscoco import load_data
from fastestimator.estimator.trace import ModelSaver
from fastestimator.network.loss import Loss
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.record.preprocess import ImageReader, Resize
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
        return (y_norm)

    def calculate_feature_recon_loss(self, y_true, y_pred):
        y_diff = y_pred - y_true
        num_elts = tf.cast(tf.reduce_prod(y_diff.shape[1:]), tf.float32)
        y_diff_norm = tf.reduce_sum(tf.square(y_diff), axis=(1, 2, 3)) / num_elts
        return (y_diff_norm)

    def calculate_gram_matrix(self, x):
        x = tf.cast(x, tf.float32)
        num_elts = tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)
        gram_matrix = tf.einsum('bijc,bijd->bcd', x, x)
        gram_matrix /= num_elts
        return gram_matrix

    def calculate_total_variation(self, y_pred):
        return (tf.image.total_variation(y_pred))

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


def get_estimator(style_img_path=None,
                  data_path=None,
                  style_weight=5.0,
                  content_weight=1.0,
                  tv_weight=1e-4,
                  model_dir=tempfile.mkdtemp()):
    train_csv, path = load_data(data_path)
    if style_img_path is None:
        style_img_path = tf.keras.utils.get_file(
            'kandinsky.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
        )
    style_img = cv2.imread(style_img_path)
    assert (style_img is not None), "Invalid style reference image"
    tfr_save_dir = os.path.join(path, 'FEdata')
    style_img = (style_img.astype(np.float32) / 127.5) / 127.5
    style_img_t = tf.convert_to_tensor(np.expand_dims(style_img, axis=0))
    writer = fe.RecordWriter(
        train_data=train_csv,
        save_dir=tfr_save_dir,
        ops=[
            ImageReader(inputs="image", parent_path=path, outputs="image"),
            Resize(inputs="image", target_size=(256, 256), outputs="image")
        ])

    pipeline = fe.Pipeline(batch_size=4, data=writer, ops=[Rescale(inputs="image", outputs="image")])

    model = FEModel(model_def=styleTransferNet,
                    model_name="style_transfer_net",
                    loss_name="loss",
                    optimizer=tf.keras.optimizers.Adam(1e-3))

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="image_out"),
        ExtractVGGFeatures(inputs=lambda: style_img_t, outputs="y_style"),
        ExtractVGGFeatures(inputs="image", outputs="y_content"),
        ExtractVGGFeatures(inputs="image_out", outputs="y_pred"),
        StyleContentLoss(style_weight=style_weight,
                         content_weight=content_weight,
                         tv_weight=tv_weight,
                         inputs=('y_pred', 'y_style', 'y_content', 'image_out'),
                         outputs='loss')
    ])
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=2,
                             traces=ModelSaver(model_name="style_transfer_net", save_dir=model_dir))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
