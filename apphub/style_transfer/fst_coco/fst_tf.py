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
import tempfile

import cv2
import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data import mscoco
from fastestimator.layers.tensorflow import InstanceNormalization, ReflectionPadding2D
from fastestimator.op.numpyop import LambdaOp
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver


class ExtractVGGFeatures(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs, outputs, mode)
        self.vgg = LossNet()

    def forward(self, data, state):
        return self.vgg(data)


class StyleContentLoss(TensorOp):
    def __init__(self, style_weight, content_weight, tv_weight, inputs, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.average_loss = average_loss

    def calculate_style_recon_loss(self, y_true, y_pred):
        y_true_gram = self.calculate_gram_matrix(y_true)
        y_pred_gram = self.calculate_gram_matrix(y_pred)
        y_diff_gram = y_pred_gram - y_true_gram
        y_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(y_diff_gram), axis=(1, 2)))
        return y_norm

    def calculate_feature_recon_loss(self, y_true, y_pred):
        y_diff = y_pred - y_true
        num_elts = tf.cast(tf.reduce_prod(y_diff.shape[1:]), tf.float32)
        y_diff_norm = tf.reduce_sum(tf.square(y_diff), axis=(1, 2, 3)) / num_elts
        return y_diff_norm

    def calculate_gram_matrix(self, x):
        x = tf.cast(x, tf.float32)
        num_elts = tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)
        gram_matrix = tf.einsum('bijc,bijd->bcd', x, x)
        gram_matrix /= num_elts
        return gram_matrix

    def calculate_total_variation(self, y_pred):
        return tf.image.total_variation(y_pred)

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
        loss = style_loss + content_loss + total_variation_reg

        if self.average_loss:
            loss = reduce_mean(loss)

        return loss


def _residual_block(x0, num_filter, kernel_size=(3, 3), strides=(1, 1)):
    initializer = tf.random_normal_initializer(0., 0.02)
    x0_cropped = tf.keras.layers.Cropping2D(cropping=2)(x0)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x0)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.Add()([x, x0_cropped])
    return x


def _conv_block(x0, num_filter, kernel_size=(9, 9), strides=(1, 1), padding="same", apply_relu=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    if apply_relu:
        x = tf.keras.layers.ReLU()(x)
    return x


def _upsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters=num_filter,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _downsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def StyleTransferNet(input_shape=(256, 256, 3), num_resblock=5):
    """Creates the Style Transfer Network.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    x = ReflectionPadding2D(padding=(40, 40))(x0)
    x = _conv_block(x, num_filter=32)
    x = _downsample(x, num_filter=64)
    x = _downsample(x, num_filter=128)

    for _ in range(num_resblock):
        x = _residual_block(x, num_filter=128)

    x = _upsample(x, num_filter=64)
    x = _upsample(x, num_filter=32)
    x = _conv_block(x, num_filter=3, apply_relu=False)
    x = tf.keras.layers.Activation("tanh")(x)
    return tf.keras.Model(inputs=x0, outputs=x)


def LossNet(input_shape=(256, 256, 3),
            style_layers=("block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"),
            content_layers=("block3_conv3",)):
    """Creates the network to compute the style loss.
    This network outputs a dictionary with outputs values for style and content, based on a list of layers from VGG16
    for each.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    mdl = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=x0)
    # Compute style loss
    style_output = [mdl.get_layer(name).output for name in style_layers]
    content_output = [mdl.get_layer(name).output for name in content_layers]
    output = {"style": style_output, "content": content_output}
    return tf.keras.Model(inputs=x0, outputs=output)


def get_estimator(batch_size=4,
                  epochs=2,
                  train_steps_per_epoch=None,
                  log_steps=100,
                  style_weight=5.0,
                  content_weight=1.0,
                  tv_weight=1e-4,
                  save_dir=tempfile.mkdtemp(),
                  style_img_path='Vassily_Kandinsky,_1913_-_Composition_7.jpg',
                  data_dir=None):
    train_data, _ = mscoco.load_data(root_dir=data_dir, load_bboxes=False, load_masks=False, load_captions=False)

    style_img = cv2.imread(style_img_path)
    assert style_img is not None, "cannot load the style image, please go to the folder with style image"
    style_img = cv2.resize(style_img, (256, 256))
    style_img = (style_img.astype(np.float32) - 127.5) / 127.5

    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
            Resize(height=256, width=256, image_in="image", image_out="image"),
            LambdaOp(fn=lambda: style_img, outputs="style_image"),
        ])

    model = fe.build(model_fn=StyleTransferNet,
                     model_name="style_transfer_net",
                     optimizer_fn=lambda: tf.optimizers.Adam(1e-3))

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="image_out"),
        ExtractVGGFeatures(inputs="style_image", outputs="y_style"),
        ExtractVGGFeatures(inputs="image", outputs="y_content"),
        ExtractVGGFeatures(inputs="image_out", outputs="y_pred"),
        StyleContentLoss(style_weight=style_weight,
                         content_weight=content_weight,
                         tv_weight=tv_weight,
                         inputs=('y_pred', 'y_style', 'y_content', 'image_out'),
                         outputs='loss'),
        UpdateOp(model=model, loss_name="loss")
    ])

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=ModelSaver(model=model, save_dir=save_dir, frequency=1),
                             epochs=epochs,
                             train_steps_per_epoch=train_steps_per_epoch,
                             log_steps=log_steps)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
