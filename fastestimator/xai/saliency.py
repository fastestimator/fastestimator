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
# noinspection PyPackageRequirements
import tensorflow as tf

from fastestimator.xai.util import show_image, show_text, show_gray_image, GradientSaliency, \
    IntegratedGradients
from fastestimator.util.util import decode_predictions


@tf.function
def compute_percentile(tensor, percentile, keepdims=True):
    """
    Args:
        tensor: A tensor with batches on the zero axis. Shape (batch X ...)
        percentile: The percentile value to be computed for each batch in the tensor
        keepdims: Whether to keep shape compatibility with the input tensor
    Returns:
        A tensor corresponding to the given percentile value within each batch of the input tensor
    """
    result = tf.reduce_min(
        tf.math.top_k(tf.reshape(tensor, (tensor.shape[0], -1)),
                      tf.cast(tf.math.ceil((1 - percentile / 100) * tensor.shape[1] * tensor.shape[2]), tf.int32),
                      sorted=False).values,
        axis=1)
    if keepdims:
        result = tf.reshape(result, [tensor.shape[0]] + [1 for _ in tensor.shape[1:]])
    return result


@tf.function
def convert_for_visualization(batched_masks, percentile=99):
    """
    Args:
        batched_masks: Input masks, channel values to be reduced by absolute value summation
        percentile: The percentile [0-100] used to set the max value of the image
    Returns:
        A (batch X width X height) image after visualization clipping is applied
    """
    flattened_mask = tf.reduce_sum(tf.abs(batched_masks), axis=3)

    vmax = compute_percentile(flattened_mask, percentile)
    vmin = tf.reduce_min(flattened_mask, axis=(1, 2), keepdims=True)

    return tf.clip_by_value((flattened_mask - vmin) / (vmax - vmin), 0, 1)


def plot_saliency(model, model_input, baseline_input=None, decode_dictionary=None, color_map="inferno", smooth=7):
    """Displays or saves a saliency mask interpretation of the given input

    Args:
        model: A model to evaluate. Should be a classifier which takes the 0th axis as the batch axis
        model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
        baseline_input: An example of what a blank model input would be.
                        Should be a tensor with the same shape as model_input
        decode_dictionary: A dictionary of "class_idx" -> "class_name" associations
        color_map: The color map to use to visualize the saliency maps.
                        Consider "Greys_r", "plasma", or "magma" as alternatives
        smooth: The number of samples to use when generating a smoothed image
    """
    predictions = np.asarray(model(model_input))
    decoded = decode_predictions(predictions, top=3, dictionary=decode_dictionary)

    grad_sal = GradientSaliency(model)
    grad_int = IntegratedGradients(model)

    vanilla_masks = grad_sal.get_mask(model_input)
    vanilla_ims = convert_for_visualization(vanilla_masks)
    smooth_masks = grad_sal.get_smoothed_mask(model_input, nsamples=smooth)
    smooth_ims = convert_for_visualization(smooth_masks)
    smooth_integrated_masks = grad_int.get_smoothed_mask(model_input, nsamples=smooth, input_baseline=baseline_input)
    smooth_integrated_ims = convert_for_visualization(smooth_integrated_masks)

    filtered_inputs = (model_input + 1) * np.asarray(smooth_integrated_ims)[:, :, :, None] - 1

    num_rows = model_input.shape[0]
    num_cols = 6
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
        show_image(filtered_inputs[i], axis=axs[i][2], title="Filtered" if i == 0 else None)
        show_gray_image(vanilla_ims[i], axis=axs[i][3], color_map=color_map, title="Vanilla" if i == 0 else None)
        show_gray_image(smooth_ims[i], axis=axs[i][4], color_map=color_map, title="Smoothed" if i == 0 else None)
        show_gray_image(smooth_integrated_ims[i],
                        axis=axs[i][5],
                        color_map=color_map,
                        title="Integrated Smoothed" if i == 0 else None)
    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0.03, wspace=0.03)
    # plt.tight_layout(pad=0.3, h_pad=0.03, w_pad=0.03, rect=(0, 0, 0.98, 0.98))
    return fig


def visualize_saliency(model,
                       model_input,
                       baseline_input=None,
                       decode_dictionary=None,
                       color_map="inferno",
                       smooth=7,
                       save_path='.'):
    """Displays or saves a saliency mask interpretation of the given input

    Args:
        model: A model to evaluate. Should be a classifier which takes the 0th axis as the batch axis
        model_input: Input tensor, shaped for the model ex. (1, 299, 299, 3)
        baseline_input: An example of what a blank model input would be.
                        Should be a tensor with the same shape as model_input
        decode_dictionary: A dictionary of "class_idx" -> "class_name" associations
        color_map: The color map to use to visualize the saliency maps.
                        Consider "Greys_r", "plasma", or "magma" as alternatives
        smooth: The number of samples to use when generating a smoothed image
        save_path: Where to save the output, or None to display the output
    """
    plot_saliency(model=model,
                  model_input=model_input,
                  baseline_input=baseline_input,
                  decode_dictionary=decode_dictionary,
                  color_map=color_map,
                  smooth=smooth)

    if save_path is None:
        plt.show()
    else:
        save_path = os.path.dirname(save_path)
        if save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'saliency.png')
        print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
