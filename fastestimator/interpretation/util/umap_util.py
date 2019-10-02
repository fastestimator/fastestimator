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
from tqdm import trange


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
    def __init__(self, root_path, layers):
        self.root_path = root_path
        print("Saving cache files to {}".format(self.root_path))
        os.makedirs(self.root_path, exist_ok=True)
        self.idx = 0
        self.layers = layers
        self.num_layers = len(self.layers)

    def save(self, data, classes):
        if len(data) != self.num_layers:
            raise IndexError("Inconsistent Layer Count Detected")
        [
            np.save(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], self.idx)),
                    data[layer]) for layer in range(self.num_layers)
        ]
        np.save(os.path.join(self.root_path, "class{}.npy".format(self.idx)), classes)
        self.idx += 1

    def batch_cached(self, batch_id):
        return os.path.isfile(os.path.join(self.root_path, "class{}.npy".format(batch_id))) and all([
            os.path.isfile(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], batch_id)))
            for layer in range(self.num_layers)
        ])

    def load(self, batches=None):
        if batches is None:
            batches = self.idx
        data = [None for _ in range(self.num_layers)]
        classes = []

        if batches == 0:
            return

        for layer in trange(self.num_layers, desc='Loading Cache', unit='layer'):
            layer_data = []
            for batch in trange(batches, desc='Loading Cache', unit='batch', leave=False):
                dat = np.load(os.path.join(self.root_path, "layer{}-batch{}.npy".format(self.layers[layer], batch)),
                              allow_pickle=True)
                layer_data.append(dat)
            data[layer] = np.concatenate(layer_data, axis=0)

        for batch in range(batches):
            clazz = np.load(os.path.join(self.root_path, "class{}.npy".format(batch)), allow_pickle=True)
            classes.extend(clazz)

        return data, classes
