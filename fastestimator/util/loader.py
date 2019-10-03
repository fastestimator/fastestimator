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

# noinspection PyPackageRequirements
import tensorflow as tf

from fastestimator.util.util import load_image


class PathLoader(object):
    """
    Args:
        root_path: The path the the root directory containing files to be read
        batch: The batch size to use when loading paths. Must be positive
        input_extension: A file extension to limit what sorts of paths are returned
        recursive_search: Whether to search within subdirectories for files
    """
    def __init__(self, root_path, batch=10, input_extension=None, recursive_search=True):

        if not os.path.isdir(root_path):
            raise AssertionError("Provided path is not a directory")
        self.root_path = root_path
        if batch < 1:
            raise AssertionError("Batch size must be positive")
        self.batch = batch
        self.input_extension = input_extension
        self.recursive_search = recursive_search
        self.current_idx = 0
        self.path_pairs = self.get_file_paths()

    def __iter__(self):
        return self

    def get_file_paths(self):
        path_pairs = []
        for root, dirs, files in os.walk(self.root_path):
            for file_name in files:
                if file_name.startswith(".") or (self.input_extension is not None
                                                 and not file_name.endswith(self.input_extension)):
                    continue
                path_pairs.append((os.path.join(root, file_name), os.path.basename(root)))
            if not self.recursive_search:
                break
        return path_pairs

    def __next__(self):
        if self.current_idx >= len(self.path_pairs):
            raise StopIteration
        else:
            result = self.path_pairs[self.current_idx:self.current_idx + self.batch]
            self.current_idx += self.batch
            return result

    def __len__(self):
        return math.ceil(len(self.path_pairs) / self.batch)


class ImageLoader(PathLoader):
    def __init__(self, root_path, model, batch=10, input_extension=None, strip_alpha=False):
        super().__init__(root_path, batch, input_extension)
        self.strip_alpha = strip_alpha
        self.input_type = model.input.dtype
        self.input_shape = model.input.shape
        if not (3 <= len(self.input_shape) <= 4):
            raise AssertionError("Model must have 3 or 4 dimensions: (batch, x, y, [channels])")
        if self.input_shape[0] is not None:
            raise AssertionError("Model must take batch on axis zero")
        self.n_channels = 0 if len(self.input_shape) == 3 else self.input_shape[3]

    def __next__(self):
        paths = super().__next__()
        inputs = [
            load_image(paths[i][0], strip_alpha=self.strip_alpha, channels=self.n_channels) for i in range(len(paths))
        ]
        batch_inputs = tf.stack([
            tf.image.resize_with_crop_or_pad(tf.convert_to_tensor(im, dtype=self.input_type), self.input_shape[1],
                                             self.input_shape[2]) for im in inputs
        ], axis=0)

        batch_classes = [paths[i][1] for i in range(len(paths))]
        return batch_inputs, batch_classes
