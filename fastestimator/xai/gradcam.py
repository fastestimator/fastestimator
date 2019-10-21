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
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tf_explain.utils.display import heatmap_display

from fastestimator.xai.util import show_image, show_text
from fastestimator.util.util import decode_predictions


class FEGradCAM(GradCAM):
    def explain(self, model_input, model, layer_name, class_index, colormap=cv2.COLORMAP_INFERNO):
        """
        Compute GradCAM for a specific class index.

        Args:
            model_input (tf.tensor): Data to perform the evaluation on.
            model (tf.keras.Model): tf.keras model to inspect
            layer_name (str): Targeted layer for GradCAM
            class_index (int, None): Index of targeted class
            colormap (int): Used in parent method signature, but ignored here

        Returns:
            tf.cams: The gradcams
        """
        outputs, guided_grads, predictions = FEGradCAM.get_gradients_and_filters(model, model_input, layer_name,
                                                                                 class_index)
        cams = GradCAM.generate_ponderated_output(outputs, guided_grads)

        input_min = tf.reduce_min(model_input)
        input_max = tf.reduce_max(model_input)

        # Need to move input image into the 0-255 range
        adjust_sum = 0.0
        adjust_factor = 1.0
        if input_min < 0:
            adjust_sum = 1.0
            adjust_factor /= 2.0
        if input_max <= 1:
            adjust_factor *= 255.0

        heatmaps = [
            heatmap_display(cam.numpy(), (inp.numpy() + adjust_sum) * adjust_factor, colormap) for cam,
            inp in zip(cams, model_input)
        ]

        return heatmaps, predictions

    @staticmethod
    @tf.function
    def get_gradients_and_filters(model, images, layer_name, class_index):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (tf.tensor): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int, None): Index of targeted class. If None will explain the class the network predicted

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(images)
            if class_index is not None:
                loss = predictions[:, class_index]
            else:
                class_indices = tf.reshape(tf.argmax(predictions, 1, output_type='int64'), (images.shape[0], 1))
                row_indices = tf.reshape(tf.range(class_indices.shape[0], dtype='int64'), (class_indices.shape[0], 1))
                classes = tf.concat([row_indices, class_indices], 1)
                loss = tf.gather_nd(predictions, classes)

        grads = tape.gradient(loss, conv_outputs)

        guided_grads = (tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads)

        return conv_outputs, guided_grads, predictions


def plot_gradcam(inputs, model, layer_id=None, target_class=None, decode_dictionary=None,
                 colormap=cv2.COLORMAP_INFERNO):
    """Creates a GradCam interpretation of the given input

    Args:
        inputs (tf.tensor): Model input, with batch along the zero axis
        model (tf.keras.model): tf.keras model to inspect
        layer_id (int, str, None): Which layer to inspect. Should be a convolutional layer. If None, the last \
                                    acceptable layer from the model will be selected
        target_class (int, None): Which output class to try to explain. None will default to explaining the maximum \
                                    likelihood prediction
        decode_dictionary (dict): A dictionary of "class_idx" -> "class_name" associations
        colormap (int): Which colormap to use when generating the heatmaps

    Returns:
        The matplotlib figure handle
    """
    gradcam = FEGradCAM()
    if isinstance(layer_id, int):
        layer_id = model.layers[layer_id].name
    if layer_id is None:
        for layer in reversed(model.layers):
            if layer.output.shape.ndims == 4:
                layer_id = layer.name
                break

    heatmaps, predictions = gradcam.explain(model_input=inputs,
                                            model=model,
                                            layer_name=layer_id,
                                            class_index=target_class,
                                            colormap=colormap)

    decoded = decode_predictions(np.asarray(predictions), top=3, dictionary=decode_dictionary)

    num_rows = math.ceil(inputs.shape[0] / 2.0)
    num_cols = 6
    dpi = 96.0

    box_width = max(220, inputs.shape[2])
    box_height = max(220, inputs.shape[1])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * (box_width / dpi), num_rows * (box_height / dpi)),
                            dpi=dpi)
    if num_rows == 1:
        axs = [axs]  # axs object not wrapped if there's only one row

    odd_cols = inputs.shape[0] % 2 == 1
    if odd_cols:
        axs[num_rows - 1][3].axis('off')
        axs[num_rows - 1][4].axis('off')
        axs[num_rows - 1][5].axis('off')

    for row in range(num_rows):
        for idx, cols in enumerate(((0, 1, 2), (3, 4, 5))):
            if row == num_rows - 1 and idx == 1 and odd_cols:
                break
            show_text(np.ones_like(inputs[2 * row + idx]),
                      decoded[2 * row + idx],
                      axis=axs[row][cols[0]],
                      title="Predictions" if row == 0 else None)
            show_image(inputs[2 * row + idx], axis=axs[row][cols[1]], title="Raw" if row == 0 else None)
            show_image(heatmaps[2 * row + idx], axis=axs[row][cols[2]], title="GradCam" if row == 0 else None)

    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0.03, wspace=0.03)

    return fig


def visualize_gradcam(inputs,
                      model,
                      layer_id=None,
                      target_class=None,
                      decode_dictionary=None,
                      colormap=cv2.COLORMAP_INFERNO,
                      save_path='.'):
    """Displays or saves a GradCam interpretation of the given input

    Args:
        inputs (tf.tensor): Model input, with batch along the zero axis
        model (tf.keras.model): tf.keras model to inspect
        layer_id (int, str, None): Which layer to inspect. Should be a convolutional layer. If None, the last \
                                    acceptable layer from the model will be selected
        target_class (int, None): Which output class to try to explain. None will default to explaining the maximum \
                                    likelihood prediction
        decode_dictionary (dict): A dictionary of "class_idx" -> "class_name" associations
        colormap (int): Which colormap to use when generating the heatmaps
        save_path (str, None): Where to save the image. If None then the image will be displayed instead
    """
    plot_gradcam(inputs=inputs,
                 model=model,
                 layer_id=layer_id,
                 target_class=target_class,
                 decode_dictionary=decode_dictionary,
                 colormap=colormap)
    if save_path is None:
        plt.show()
    else:
        save_path = os.path.dirname(save_path)
        if save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'gradCam.png')
        print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
