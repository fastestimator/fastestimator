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
import argparse
import os
import re

import numpy as np
# noinspection PyPackageRequirements
import tensorflow as tf
from pyfiglet import Figlet
from tensorflow import keras
from tensorflow.python import keras
from tqdm import tqdm

from fastestimator.xai import visualize_caricature, visualize_saliency, visualize_gradcam
from fastestimator.xai.umaps import UmapPlotter
from fastestimator.xai.util.umap_util import Evaluator, FileCache
from fastestimator.summary import Summary
from fastestimator.summary.logs import visualize_logs
from fastestimator.util.loader import ImageLoader, PathLoader
from fastestimator.util.util import is_number, load_image, load_dict, strip_suffix, parse_string_to_python


class SaveAction(argparse.Action):
    """
    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
    is invoked directly during argument parsing.
    """
    def __init__(self, option_strings, dest, nargs='?', **kwargs):
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))


def draw():
    print(Figlet(font="slant").renderText("FastEstimator"))


def parse_cli_to_dictionary(input_list):
    """
    Args:
        input_list: A list of input strings from a cli

    Returns:
        A dictionary constructed from the input list, with values converted to python objects where applicable
    """
    result = {}
    if input_list is None:
        return result
    key = ""
    val = ""
    idx = 0
    while idx < len(input_list):
        if input_list[idx].startswith("--"):
            if len(key) > 0:
                result[key] = parse_string_to_python(val)
            val = ""
            key = input_list[idx].strip('--')
        else:
            val += input_list[idx]
        idx += 1
    if len(key) > 0:
        result[key] = parse_string_to_python(val)
    return result


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


def load_and_saliency(model_path,
                      input_paths,
                      baseline=-1,
                      dictionary_path=None,
                      strip_alpha=False,
                      smooth_factor=7,
                      save=False,
                      save_dir=None):
    """A helper class to load input and invoke the saliency api

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


def load_and_gradcam(model_path,
                     input_paths,
                     layer_id=None,
                     dictionary_path=None,
                     strip_alpha=False,
                     save=False,
                     save_dir=None):
    """A helper class to load input and invoke the gradcam api

    Args:
        model_path: The path the model file (str)
        input_paths: The paths to model input files [(str),...] or to a folder of inputs [(str)]
        layer_id: The layer id to be used. None defaults to the last conv layer in the model
        dictionary_path: The path to a dictionary file encoding a 'class_idx'->'class_name' mapping
        strip_alpha: Whether to collapse alpha channels when loading an input (bool)
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

    visualize_gradcam(inputs=tf_image, model=network, layer_id=layer_id, decode_dictionary=dic, save_path=save_dir)


def load_and_umap(model_path,
                  input_root_path,
                  print_layers=False,
                  strip_alpha=False,
                  layers=None,
                  input_extension=None,
                  batch=10,
                  use_cache=True,
                  cache_dir=None,
                  dictionary_path=None,
                  save=False,
                  save_dir=None,
                  legend_mode='shared',
                  umap_parameters=None):
    if umap_parameters is None:
        umap_parameters = {}
    if save is True and save_dir is None:
        save_dir = os.path.dirname(model_path)
    if cache_dir is None:
        # If the user passes the input dir as a relative path without ./ then dirname will contain all path info
        if os.path.basename(input_root_path) == "":
            cache_dir = os.path.dirname(input_root_path) + "__layer_outputs"
        else:
            cache_dir = os.path.join(os.path.dirname(input_root_path),
                                     os.path.basename(input_root_path) + "__layer_outputs")

    network = keras.models.load_model(model_path, compile=False)
    if print_layers:
        for idx, layer in enumerate(network.layers):
            print("{}: {} --- output shape: {}".format(idx, layer.name, layer.output_shape))
        return

    evaluator = Evaluator(network, layers=layers)
    loader = ImageLoader(input_root_path,
                         network,
                         batch=batch,
                         input_extension=input_extension,
                         strip_alpha=strip_alpha)
    cache = FileCache(cache_dir, evaluator.layers) if use_cache else None
    plotter = UmapPlotter(load_dict(dictionary_path, True), **umap_parameters)

    classes = []
    layer_outputs = None
    for batch_id, (batch_inputs, batch_classes) in enumerate(tqdm(loader, desc='Computing Outputs', unit='batch')):
        if use_cache and cache.batch_cached(batch_id):
            continue
        batch_layer_outputs = evaluator.evaluate(batch_inputs)
        if use_cache:
            cache.save(batch_layer_outputs, batch_classes)
        else:
            if layer_outputs is None:
                layer_outputs = batch_layer_outputs
            else:
                for i, (layer, batch_layer) in enumerate(zip(layer_outputs, batch_layer_outputs)):
                    layer_outputs[i] = np.concatenate((layer, batch_layer), axis=0)
            classes.extend(batch_classes)
    if use_cache:
        layer_outputs, classes = cache.load(len(loader))

    plotter.visualize_umap(
        layer_outputs,
        labels=classes,
        legend_loc=legend_mode,
        save_path=save_dir,
        title=[
            "Layer {}: {}".format(evaluator.layers[idx], network.layers[evaluator.layers[idx]].name)
            for idx in range(len(layer_outputs))
        ])


def _parse_file(file_path, file_extension):
    """ A function which will parse log files into a dictionary of metrics

    Args:
        file_path (str): The path to a log file
        file_extension (str): The extension of the log file
    Returns:
        An experiment summarizing the given log file
    """
    # TODO: need to handle multi-line output like confusion matrix
    experiment = Summary(strip_suffix(os.path.split(file_path)[1].strip(), file_extension))
    with open(file_path) as file:
        for line in file:
            mode = None
            if line.startswith("FastEstimator-Train"):
                mode = "train"
            elif line.startswith("FastEstimator-Eval"):
                mode = "eval"
            if mode is None:
                continue
            parsed_line = re.findall(r"([^:^;\s]+):[\s]*([-]?[0-9]+[.]?[0-9]*);", line)
            step = parsed_line[0]
            assert step[0] == "step", \
                "Log file (%s) seems to be missing step information, or step is not listed first" % file
            for metric in parsed_line[1:]:
                experiment.history[mode][metric[0]].update({int(step[1]): float(metric[1])})
    return experiment


def parse_log_files(file_paths,
                    log_extension='.txt',
                    smooth_factor=0,
                    save=False,
                    save_path=None,
                    ignore_metrics=None,
                    share_legend=True,
                    pretty_names=False):
    """A function which will iterate through the given log file paths, parse them to extract metrics, remove any
    metrics which are blacklisted, and then pass the necessary information on the graphing function

    Args:
        file_paths: A list of paths to various log files
        log_extension: The extension of the log files
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided
        ignore_metrics: Any metrics within the log files which will not be visualized
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None
    """
    if file_paths is None or len(file_paths) < 1:
        raise AssertionError("must provide at least one log file")
    if save and save_path is None:
        save_path = file_paths[0]

    experiments = []
    for file_path in file_paths:
        experiments.append(_parse_file(file_path, log_extension))
    visualize_logs(experiments,
                   save_path=save_path,
                   smooth_factor=smooth_factor,
                   share_legend=share_legend,
                   pretty_names=pretty_names,
                   ignore_metrics=ignore_metrics)


def parse_log_dir(dir_path,
                  log_extension='.txt',
                  recursive_search=False,
                  smooth_factor=1,
                  save=False,
                  save_path=None,
                  ignore_metrics=None,
                  share_legend=True,
                  pretty_names=False):
    """ A function which will gather all log files within a given folder and pass them along for visualization

    Args:
        dir_path: The path to a directory containing log files
        log_extension: The extension of the log files
        recursive_search: Whether to recursively search sub-directories for log files
        smooth_factor: A non-negative float representing the magnitude of gaussian smoothing to apply(zero for none)
        save: Whether to save (true) or display (false) the generated graph
        save_path: Where to save the image if save is true. Defaults to dir_path if not provided
        ignore_metrics: Any metrics within the log files which will not be visualized
        share_legend: Whether to have one legend across all graphs (true) or one legend per graph (false)
        pretty_names: Whether to modify the metric names in graph titles (true) or leave them alone (false)

    Returns:
        None
    """
    loader = PathLoader(dir_path, input_extension=log_extension, recursive_search=recursive_search)
    file_paths = [x[0] for x in loader.path_pairs]

    parse_log_files(file_paths,
                    log_extension,
                    smooth_factor,
                    save,
                    save_path,
                    ignore_metrics,
                    share_legend,
                    pretty_names)
