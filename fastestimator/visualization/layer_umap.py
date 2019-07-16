import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# noinspection PyPackageRequirements
import tensorflow as tf
# noinspection PyPackageRequirements
import umap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
# noinspection PyPackageRequirements
from tensorflow.python import keras
from tqdm import tqdm, trange

from fastestimator.util.loader import ImageLoader
from fastestimator.util.util import load_dict, Suppressor

assert Axes3D  # Axes3D is used to enable projection='3d', but will show up as unused without the assert


def map_classes_to_colors(classifications):
    classes = set(classifications)
    num_classes = len(classes)
    colors = sns.hls_palette(n_colors=num_classes, s=0.95) if num_classes > 10 else sns.color_palette(
        "colorblind")
    class_to_color = {clazz: idx for idx, clazz in enumerate(classes)}
    return [colors[class_to_color[clazz]] for clazz in classifications], {clazz: colors[class_to_color[clazz]] for clazz
                                                                          in classes}


def draw_umaps(layer_outputs, classifications, dictionary=None, layers=None, layer_ids=None, save=False, save_path='.',
               legend_mode='shared'):
    color_list, color_map = map_classes_to_colors(classifications)

    num_layers = len(layer_outputs)
    n_components = len(layer_outputs[0][0])
    num_cols = math.ceil(math.sqrt(num_layers))
    num_rows = math.ceil(num_layers / num_cols)
    layer_grid_location = {idx: (idx // num_cols, idx % num_cols) for idx in range(num_layers)}

    sns.set_context('paper')
    if n_components == 3:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.8 * num_rows),
                                subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.8 * num_rows))

    # If only one row, need to re-format the axs object for consistency. Likewise for columns
    if num_rows == 1:
        axs = [axs]
        if num_cols == 1:
            axs = [axs]

    # some of the columns in the last row might be unused, so disable them
    last_column_idx = num_cols - (num_rows * num_cols - num_layers) - 1
    for i in range(last_column_idx + 1, num_cols):
        axs[num_rows - 1][i].axis('off')

    # Turn off axis since numeric values are not meaningful
    for i in range(num_rows):
        for j in range(num_cols):
            axis = axs[i][j]
            axis.set_yticks([], [])
            axis.set_yticklabels([])
            axis.set_xticks([], [])
            axis.set_xticklabels([])

    for idx, layer in enumerate(layer_outputs):
        ax = axs[layer_grid_location[idx][0]][layer_grid_location[idx][1]]
        layer_id = idx if layer_ids is None else layer_ids[idx]
        layer_name = "" if layers is None else ": " + layers[layer_id].name
        ax.set_title("Layer {}{}".format(layer_id, layer_name))
        if n_components == 1:
            ax.scatter(layer[:, 0], range(len(layer)), c=color_list, s=3)
        if n_components == 2:
            ax.scatter(layer[:, 0], layer[:, 1], c=color_list, s=3)
        if n_components == 3:
            ax.scatter(layer[:, 0], layer[:, 1], layer[:, 2], c=color_list, s=3)
    plt.tight_layout()

    if legend_mode != 'off':
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[clazz],
                                  label=clazz if dictionary is None else dictionary[clazz],
                                  markersize=7) for clazz in color_map]
        if legend_mode == 'shared' and num_rows > 1:
            if last_column_idx == num_cols - 1:
                fig.subplots_adjust(bottom=0.15)
                fig.legend(handles=legend_elements, loc='lower center', ncol=num_cols + 1)
            else:
                axs[num_rows - 1][last_column_idx + 1].legend(handles=legend_elements, loc='center', fontsize='large')
        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    if i == num_rows - 1 and j > last_column_idx:
                        break
                    axs[i][j].legend(handles=legend_elements, loc='best', fontsize='small')

    if not save:
        plt.show()
    else:
        if save_path is None or save_path == "":
            save_path = "."
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, 'umaps.png')
        print("Saving to {}".format(save_file))
        plt.savefig(save_file, dpi=300)


class Evaluator(object):
    def __init__(self, model, layers=None):
        """
        Args:
            model: The ML model to generate outputs from
            layers: The layer indices to be investigated by the evaluator. If not provided then all layers will be used
        """
        self.model = model
        self.layers = layers if layers is not None and len(layers) > 0 else [i for i in range(model.layers)]
        self.num_layers = len(self.layers)
        self.functor = tf.keras.backend.function([self.model.input], [self.model.layers[i].output for i in self.layers])

    def evaluate(self, x):
        layer_outputs = self.functor([x])
        return [np.reshape(layer, [x.shape[0], -1]) for layer in layer_outputs]


class FileCache(object):
    def __init__(self, root_path, layers, umap_parameters):
        self.root_path = root_path
        print("Saving cache files to {}".format(self.root_path))
        os.makedirs(self.root_path, exist_ok=True)
        self.idx = 0
        self.layers = layers
        self.num_layers = len(self.layers)
        self.fit = umap.UMAP(**umap_parameters)

    def save(self, data, classes):
        if len(data) != self.num_layers:
            raise IndexError("Inconsistent Layer Count Detected")
        [np.save(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], self.idx)), data[layer])
         for layer in range(self.num_layers)]
        np.save(os.path.join(self.root_path, "class{}.npy".format(self.idx)), classes)
        self.idx += 1

    def batch_cached(self, batch_id):
        return os.path.isfile(os.path.join(self.root_path, "class{}.npy".format(batch_id))) and all(
            [os.path.isfile(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], batch_id)))
             for layer in range(self.num_layers)])

    def load_and_transform(self, batches=None):
        if batches is None:
            batches = self.idx
        data = [None for _ in range(self.num_layers)]
        classes = []

        if batches == 0:
            return data, classes

        for layer in trange(self.num_layers, desc='Computing UMaps', unit='layer'):
            layer_data = []
            for batch in trange(batches, desc='Loading Cache', unit='batch', leave=False):
                dat = np.load(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], batch)),
                              allow_pickle=True)
                layer_data.append(dat)
            layer_data = np.concatenate(layer_data, axis=0)
            with Suppressor():  # Silence a bunch of numba warnings
                data[layer] = self.fit.fit_transform(layer_data)

        for batch in range(batches):
            clazz = np.load(os.path.join(self.root_path, "class{}.npy".format(batch)), allow_pickle=True)
            classes.extend(clazz)

        return data, classes


def umap_layers(model_path, input_root_path, print_layers=False, strip_alpha=False, layers=None,
                input_extension=None, batch=10, use_cache=True, cache_dir=None, dictionary_path=None, save=False,
                save_dir=None, legend_mode='shared', umap_parameters=None):
    if umap_parameters is None:
        umap_parameters = {}
    if save_dir is None:
        save_dir = os.path.dirname(model_path)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(input_root_path),
                                 os.path.basename(input_root_path) + "__layer_outputs")

    network = keras.models.load_model(model_path)
    if print_layers:
        for idx, layer in enumerate(network.layers):
            print("{}: {} --- output shape: {}".format(idx, layer.name, layer.output_shape))
        return

    evaluator = Evaluator(network, layers=layers)
    loader = ImageLoader(input_root_path, network, batch=batch, input_extension=input_extension,
                         strip_alpha=strip_alpha)
    cache = FileCache(cache_dir, evaluator.layers, umap_parameters) if use_cache else None

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
        layer_outputs, classes = cache.load_and_transform(len(loader))
    else:
        fit = umap.UMAP(**umap_parameters)
        with Suppressor():  # Silence a bunch of numba warnings
            layer_outputs = [fit.fit_transform(layer) for layer in layer_outputs]
    draw_umaps(layer_outputs, classes, layer_ids=layers, layers=network.layers, save=save, save_path=save_dir,
               dictionary=load_dict(dictionary_path, True),  legend_mode=legend_mode)
