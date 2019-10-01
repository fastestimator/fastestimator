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

import numpy as np
# noinspection PyPackageRequirements
import tensorflow as tf
# noinspection PyPackageRequirements
from tensorflow.python import keras

from fastestimator.util.loader import PathLoader
from fastestimator.util.util import is_number, load_dict, load_image
from fastestimator.visualization.saliencies import visualize_saliency


def load_and_saliency(model_path,
                      input_paths,
                      baseline=-1,
                      dictionary_path=None,
                      strip_alpha=False,
                      smooth_factor=7,
                      save=False,
                      save_dir=None):
    """ A helper class to load input and invoke the saliency api
    Args:
        model_path: The path the model file (str)
        input_paths: The paths to model input files [(str),...] or to a folder of inputs [(str)]
        baseline: Either a number corresponding to the baseline for integration, or a path to a baseline file
        dictionary_path: The path to a dictionary file encoding a 'class_idx'->'class_name' mapping
        strip_alpha: Whether to collapse alpha channels when loading an input (bool)
        smooth_factor: How many iterations of the smoothing algorithm to run (int)
        save: Whether to save (True) or display (False) the resulting image
        save_dir: Where to save the image if save=True
    """
    model_dir = os.path.dirname(model_path)
    if save_dir is None:
        save_dir = model_dir
    if not save:
        save_dir = None
    network = keras.models.load_model(model_path, compile=False)
    input_type = network.input.dtype
    input_shape = network.input.shape
    n_channels = 0 if len(input_shape) == 3 else input_shape[3]

    dic = load_dict(dictionary_path)
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        loader = PathLoader(input_paths[0])
        input_paths = [path[0] for path in loader.path_pairs]
    inputs = [load_image(input_paths[i], strip_alpha=strip_alpha, channels=n_channels) for i in range(len(input_paths))]
    max_shapes = np.maximum.reduce([inp.shape for inp in inputs], axis=0)
    tf_image = tf.stack([
        tf.image.resize_with_crop_or_pad(tf.convert_to_tensor(im, dtype=input_type), max_shapes[0], max_shapes[1])
        for im in inputs
    ],
                        axis=0)
    if is_number(baseline):
        baseline_gen = tf.constant_initializer(float(baseline))
        baseline_image = baseline_gen(shape=tf_image.shape, dtype=input_type)
    else:
        baseline_image = load_image(baseline)
        baseline_image = tf.convert_to_tensor(baseline_image, dtype=input_type)

    visualize_saliency(network,
                       tf_image,
                       baseline_input=baseline_image,
                       decode_dictionary=dic,
                       smooth=smooth_factor,
                       save_path=save_dir)
