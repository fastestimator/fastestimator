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

import tensorflow as tf
from tensorflow import keras

from fastestimator.util.loader import PathLoader
from fastestimator.util.util import load_dict, load_image
from fastestimator.visualization.caricatures import visualize_caricature


def load_and_caricature(model_path,
                        input_paths,
                        dictionary_path=None,
                        save=False,
                        save_dir=None,
                        strip_alpha=False,
                        layer_ids=None,
                        print_layers=False,
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
        model_path (str): The path to a keras model to be inspected by the Caricature visualization
        layer_ids (int, list): The layer(s) of the model to be inspected by the Caricature visualization
        input_paths (list): Strings corresponding to image files to be visualized
        dictionary_path (string): A path to a dictionary mapping model outputs to class names
        save (bool): Whether to save (True) or display (False) the result
        save_dir (str): Where to save the image if save is True
        strip_alpha (bool): Whether to strip the alpha channel from input images
        print_layers (bool): Whether to skip visualization and instead just print out the available layers in a model \
                            (useful for deciding which layers you might want to caricature)
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
    model_dir = os.path.dirname(model_path)
    if save_dir is None and save:
        save_dir = model_dir
    network = keras.models.load_model(model_path, compile=False)
    input_type = network.input.dtype
    input_shape = network.input.shape
    n_channels = 0 if len(input_shape) == 3 else input_shape[3]
    input_height = input_shape[1] or 224  # If the model doesn't specify width and height, just guess 224
    input_width = input_shape[2] or 224

    if print_layers:
        for idx, layer in enumerate(network.layers):
            print("{}: {} --- output shape: {}".format(idx, layer.name, layer.output_shape))
        return

    dic = load_dict(dictionary_path)
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        loader = PathLoader(input_paths[0])
        input_paths = [path[0] for path in loader.path_pairs]
    inputs = [load_image(input_paths[i], strip_alpha=strip_alpha, channels=n_channels) for i in range(len(input_paths))]
    tf_image = tf.stack([
        tf.image.resize_with_pad(tf.convert_to_tensor(im, dtype=input_type),
                                 input_height,
                                 input_width,
                                 method='lanczos3') for im in inputs
    ])
    tf_image = tf.clip_by_value(tf_image, -1, 1)

    visualize_caricature(network,
                         tf_image,
                         layer_ids=layer_ids,
                         decode_dictionary=dic,
                         save_path=save_dir,
                         n_steps=n_steps,
                         learning_rate=learning_rate,
                         blur=blur,
                         cossim_pow=cossim_pow,
                         sd=sd,
                         fft=fft,
                         decorrelate=decorrelate,
                         sigmoid=sigmoid)
