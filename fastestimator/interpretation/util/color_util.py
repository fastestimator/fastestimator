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
import numpy as np
import tensorflow as tf

# Empirical (ImageNet) color correlation matrix, from https://github.com/tensorflow/lucid
# yapf: disable
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
# yapf: enable
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
color_correlation_normalized_transpose = color_correlation_normalized.T

color_mean = [0.48, 0.46, 0.41]


@tf.function
def linear_decorelate_color(image):
    """Multiply input by sqrt of empirical color correlation matrix.

    If you interpret an image's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """
    if image.shape[-1] == 3:
        t_flat = tf.reshape(image, [-1, 3])
        t_flat = tf.matmul(t_flat, color_correlation_normalized_transpose)
        image = tf.reshape(t_flat, tf.shape(image))
    return image


@tf.function
def to_valid_rgb(image, decorrelate=False, sigmoid=True):
    """Transform inner dimension of image to valid rgb colors.

    In practice this consists of two parts:
    (1) If requested, transform the colors from a decorrelated color space to RGB.
    (2) Constrain the color channels to be in [-1,1], either using a sigmoid function or clipping.

    Args:
        image: input tensor, innermost dimension will be interpreted as colors and transformed/constrained.
        decorrelate: should the input tensor's colors be interpreted as coming from a whitened space or not?
        sigmoid: should the colors be constrained using sigmoid (if True) or clipping (if False).
    Returns:
      'image' with the innermost dimension transformed.
    """
    # TODO let the user specify / infer the data range instead of forcing [-1,1]
    if decorrelate:
        image = linear_decorelate_color(image)
    if decorrelate and not sigmoid:
        image += color_mean
    if sigmoid:
        return 2.0 * tf.nn.sigmoid(image) - 1.0
    else:
        return tf.clip_by_value(image, -1.0, 1.0)
