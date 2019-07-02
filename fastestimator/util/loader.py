import math
import os

import numpy as np
# noinspection PyPackageRequirements
import tensorflow as tf

from fastestimator.util.util import load_image


class PathLoader(object):
    def __init__(self, root_path, batch=10, input_extension=None):
        """
        Args:
            root_path: The path the the root directory containing files to be read
        """
        if not os.path.isdir(root_path):
            raise AssertionError("Provided path is not a directory")
        self.root_path = root_path
        if batch < 1:
            raise AssertionError("Batch size must be positive")
        self.batch = batch
        self.input_extension = input_extension
        self.current_idx = 0
        self.path_pairs = self.get_file_paths()

    def __iter__(self):
        return self

    def get_file_paths(self):
        path_pairs = []
        for root, dirs, files in os.walk(self.root_path):
            for file_name in files:
                if file_name.startswith(".") or (
                        self.input_extension is not None and not file_name.endswith(self.input_extension)):
                    continue
                path_pairs.append((os.path.join(root, file_name), os.path.basename(root)))
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
    def __init__(self, root_path, batch=10, input_extension=None, strip_alpha=False, input_type='float32'):
        super(ImageLoader, self).__init__(root_path, batch, input_extension)
        self.strip_alpha = strip_alpha
        self.input_type = input_type

    def __next__(self):
        paths = super(ImageLoader, self).__next__()
        inputs = [load_image(paths[i][0], strip_alpha=self.strip_alpha) for i in range(len(paths))]
        max_shapes = np.maximum.reduce([inp.shape for inp in inputs], axis=0)
        max_shapes[
            0] = 500  # problem: if different batches have different im size then the feature count will be different
        max_shapes[1] = 500
        batch_inputs = tf.stack([tf.image.resize_with_crop_or_pad(tf.convert_to_tensor(im, dtype=self.input_type),
                                                                  max_shapes[0], max_shapes[1]) for im in inputs],
                                axis=0)

        batch_classes = [paths[i][1] for i in range(len(paths))]
        return batch_inputs, batch_classes
