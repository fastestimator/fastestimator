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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange

from fastestimator.xai.util import to_valid_rgb, blur_image, fft_vars_to_whitened_im, rfft2d_freqs, \
    show_image, show_text
from fastestimator.op.tensorop import Augmentation2D
from fastestimator.util.util import decode_predictions


@tf.function
def _dot_loss(true, pred, cossim_pow):
    reduce_axis = tf.range(1, len(true.shape))  # Reductions should keep the batch axis (zero) separated
    # reduce_axis = tf.range(len(true.shape))  # TODO - it seems empirically like this doesn't matter. why???
    dot = tf.reduce_sum(true * pred, axis=reduce_axis)
    mag = tf.sqrt(tf.reduce_sum(true**2, axis=reduce_axis)) * tf.sqrt(tf.reduce_sum(pred**2, axis=reduce_axis))
    cossim = dot / (1e-6 + mag)
    cossim = tf.maximum(0.1, cossim)
    result = dot * cossim**cossim_pow
    # Negative value since we actually want to maximize this objective
    return -1. * result


@tf.function
def _compute_gradients(model, aug, target_input, source_input, cossim_pow, decorrelate, sigmoid, fft, scale):
    aug.setup()
    transformed = aug.forward(target_input, {})
    target = model(transformed)
    with tf.GradientTape() as tape:
        transformed = source_input
        if fft:
            transformed = fft_vars_to_whitened_im(transformed, scale, target_input.shape[2])
        transformed = to_valid_rgb(transformed, decorrelate=decorrelate, sigmoid=sigmoid)
        transformed = aug.forward(transformed, {})
        layer_outputs = model(transformed)
        loss = _dot_loss(target, layer_outputs, cossim_pow)
    return tape.gradient(loss, source_input)


def _generate_variable_image(model_input, sd=0.01, fft=False, decay_power=1.0):
    sd = sd or 0.01
    if not fft:
        return tf.Variable(tf.random.normal(shape=model_input.shape, mean=tf.reduce_mean(model_input), stddev=sd),
                           trainable=True), None
    batch, h, w, ch = model_input.shape
    init_val_size = (2, batch, ch, h, w // 2 + 1 + (w % 2))  # zeroth index is for real/imaginary
    caricatures_freq = tf.Variable(tf.random.normal(shape=init_val_size, mean=0, stddev=sd))

    freqs = rfft2d_freqs(h, w)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))**decay_power
    scale *= np.sqrt(w * h)
    big_kernel = tf.eye(num_rows=scale.shape[0], num_columns=scale.shape[0], batch_shape=[batch, ch], dtype=scale.dtype)
    scale = tf.matmul(big_kernel, scale)
    scale = tf.cast(scale, tf.complex64)
    return caricatures_freq, scale


def generate_caricatures(model,
                         model_input,
                         layer_id,
                         n_steps=100,
                         learning_rate=0.05,
                         blur=1,
                         cossim_pow=0.5,
                         sd=0.01,
                         fft=True,
                         decorrelate=True,
                         sigmoid=True):
    """
    Args:
        model (model): The keras model to be inspected by the Caricature visualization
        model_input (tensor): The input images to be fed to the model
        layer_id (int): The layer of the model to be inspected by the Caricature visualization
        n_steps (int): How many steps of optimization to run when computing caricatures (quality vs time trade)
        learning_rate (float): The learning rate of the caricature optimizer. Should be higher than usual
        blur (float): How much blur to add to images during caricature generation
        cossim_pow (float): How much should similarity in form be valued versus creative license
        sd (float): The standard deviation of the noise used to seed the caricature
        fft (bool): Whether to use fft space (True) or image space (False) to create caricatures
        decorrelate (bool): Whether to use an ImageNet-derived color correlation matrix to de-correlate
                            colors in the caricature. Parameter has no effect on grey scale images.
        sigmoid (bool): Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values
    """
    small_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_id].output)

    caricatures, scale = _generate_variable_image(model_input, sd, fft, blur)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

    h = model_input.shape[1]
    w = model_input.shape[2]
    aug = Augmentation2D(rotation_range=5,
                         zoom_range=0.05,
                         shear_range=5,
                         width_shift_range=5 / w,
                         height_shift_range=5 / h)
    aug.height = h
    aug.width = w

    for i in trange(n_steps, desc="Layer {}".format(layer_id)):
        gradients = _compute_gradients(small_model,
                                       aug,
                                       model_input,
                                       caricatures,
                                       cossim_pow,
                                       decorrelate,
                                       sigmoid,
                                       fft,
                                       scale)
        # Optimizer call cannot be in tf.function b/c calling it on a newly instantiated object generates variables:
        # https://github.com/tensorflow/tensorflow/issues/27120
        optimizer.apply_gradients(zip([gradients], [caricatures]))
        if not fft:
            caricatures.assign(blur_image(caricatures, tf.constant(blur - blur * (i / n_steps))))

    if fft:
        caricatures = fft_vars_to_whitened_im(caricatures, scale, w)
    return to_valid_rgb(caricatures, decorrelate=decorrelate, sigmoid=sigmoid)


def plot_caricature(model,
                    model_input,
                    layer_ids=None,
                    decode_dictionary=None,
                    n_steps=512,
                    learning_rate=0.05,
                    blur=1,
                    cossim_pow=0.5,
                    sd=0.01,
                    fft=True,
                    decorrelate=True,
                    sigmoid=True):
    """
    Args:
        model (model): The keras model to be inspected by the Caricature visualization
        model_input (tensor): The input images to be fed to the model
        layer_ids (list): The layers of the model to be inspected by the Caricature visualization
        decode_dictionary (dict): A dictionary mapping model outputs to class names
        n_steps (int): How many steps of optimization to run when computing caricatures (quality vs time trade)
        learning_rate (float): The learning rate of the caricature optimizer. Should be higher than usual
        blur (float): How much blur to add to images during caricature generation
        cossim_pow (float): How much should similarity in form be valued versus creative license
        sd (float): The standard deviation of the noise used to seed the caricature
        fft (bool): Whether to use fft space (True) or image space (False) to create caricatures
        decorrelate (bool): Whether to use an ImageNet-derived color correlation matrix to de-correlate
                            colors in the caricature. Parameter has no effect on grey scale images.
        sigmoid (bool): Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values
    """
    if layer_ids is None or len(layer_ids) == 0:
        layer_ids = [i for i in range(len(model.layers))]

    predictions = np.asarray(model.predict(model_input))
    decoded = decode_predictions(predictions, top=3, dictionary=decode_dictionary)

    caricatures = [
        generate_caricatures(model,
                             model_input,
                             layer,
                             n_steps=n_steps,
                             learning_rate=learning_rate,
                             blur=blur,
                             cossim_pow=cossim_pow,
                             sd=sd,
                             fft=fft,
                             decorrelate=decorrelate,
                             sigmoid=sigmoid) for layer in layer_ids
    ]

    num_rows = model_input.shape[0]
    num_cols = len(layer_ids) + 2
    dpi = 96.0

    box_width = max(220, model_input.shape[2])
    box_height = max(220, model_input.shape[1])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * (box_width / dpi), num_rows * (box_height / dpi)),
                            dpi=dpi)

    if num_rows == 1:
        axs = [axs]  # axs object not wrapped if there's only one row
    for i in range(num_rows):
        show_text(np.ones_like(model_input[i]), decoded[i], axis=axs[i][0], title="Predictions" if i == 0 else None)
        show_image(model_input[i], axis=axs[i][1], title="Raw" if i == 0 else None)
        for j in range(len(layer_ids)):
            layer_id = layer_ids[j]
            layer_name = ": " + model.layers[layer_id].name
            show_image(caricatures[j][i],
                       axis=axs[i][2 + j],
                       title="Layer {}{}".format(layer_id, layer_name) if i == 0 else None)

    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0.03, wspace=0.03)
    return fig


def visualize_caricature(model,
                         model_input,
                         layer_ids=None,
                         decode_dictionary=None,
                         save_path=".",
                         n_steps=512,
                         learning_rate=0.05,
                         blur=1,
                         cossim_pow=0.5,
                         sd=0.01,
                         fft=True,
                         decorrelate=True,
                         sigmoid=True):
    """
    Args:
        model (model): The keras model to be inspected by the Caricature visualization
        model_input (tensor): The input images to be fed to the model
        layer_ids (list): The layers of the model to be inspected by the Caricature visualization
        decode_dictionary (dict): A dictionary mapping model outputs to class names
        save_path (str): The directory into which to save the caricature
        n_steps (int): How many steps of optimization to run when computing caricatures (quality vs time trade)
        learning_rate (float): The learning rate of the caricature optimizer. Should be higher than usual
        blur (float): How much blur to add to images during caricature generation
        cossim_pow (float): How much should similarity in form be valued versus creative license
        sd (float): The standard deviation of the noise used to seed the caricature
        fft (bool): Whether to use fft space (True) or image space (False) to create caricatures
        decorrelate (bool): Whether to use an ImageNet-derived color correlation matrix to de-correlate
                            colors in the caricature. Parameter has no effect on grey scale images.
        sigmoid (bool): Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values
    """
    plot_caricature(model,
                    model_input=model_input,
                    layer_ids=layer_ids,
                    decode_dictionary=decode_dictionary,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    blur=blur,
                    cossim_pow=cossim_pow,
                    sd=sd,
                    fft=fft,
                    decorrelate=decorrelate,
                    sigmoid=sigmoid)
    if save_path is None:
        plt.show()
    else:
        save_path = os.path.dirname(save_path)
        if save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'caricatures.png')
        print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
