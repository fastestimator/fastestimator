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
import tensorflow as tf

from fastestimator.op import TensorOp


class Augmentation3D(TensorOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)

        self.isometric_options = tf.convert_to_tensor([[1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0,
                                                                                                1], [0, 0, 0, 1, 1, 1],
                                                       [0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0,
                                                                                                0], [0, 1, 1, 1, 1, 0],
                                                       [0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1,
                                                                                                1], [1, 1, 1, 0, 0, 1],
                                                       [1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1,
                                                                                                1], [1, 1, 1, 0, 1, 1],
                                                       [1, 1, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0,
                                                                                                1], [1, 1, 0, 0, 0, 0],
                                                       [1, 1, 0, 1, 1,
                                                        0], [0, 0, 0, 1, 1,
                                                             0], [0, 0, 0, 0, 0,
                                                                  0], [0, 1, 0, 1, 0,
                                                                       0], [0, 1, 0, 0, 1,
                                                                            0], [1, 1, 0, 0, 1,
                                                                                 0], [0, 0, 0, 0, 1,
                                                                                      0], [0, 0, 0, 1, 0,
                                                                                           0], [0, 1, 0, 0, 0,
                                                                                                0], [0, 1, 0, 1, 1, 0],
                                                       [1, 1, 0, 1, 0,
                                                        0], [0, 1, 1, 0, 1,
                                                             1], [0, 1, 1, 1, 0,
                                                                  1], [0, 1, 1, 0, 0,
                                                                       1], [0, 1, 1, 1, 1,
                                                                            1], [1, 1, 1, 0, 1,
                                                                                 0], [1, 1, 1, 1, 0,
                                                                                      0], [0, 0, 1, 0, 1,
                                                                                           0], [0, 0, 1, 1, 0,
                                                                                                0], [1, 1, 1, 0, 0, 0],
                                                       [1, 1, 1, 1, 1,
                                                        0], [0, 0, 1, 0, 0,
                                                             0], [0, 0, 1, 1, 1,
                                                                  0], [0, 1, 0, 1, 1,
                                                                       1], [1, 1, 0, 0, 1,
                                                                            1], [0, 0, 0, 1, 0,
                                                                                 1], [1, 1, 0, 1, 0,
                                                                                      1], [0, 1, 0, 0, 0, 1]])

    # equivalent of  np.rot90(data, rotate_y, axes=(0, 2))
    def _rotate_y(self, data):
        data = tf.reverse(data, axis=[2])
        data = tf.transpose(data, perm=[2, 1, 0])
        return data

    #equivalent of  np.rot90(data, rotate_z, axes=(0, 1))
    def _rotate_z(self, data):
        data = tf.reverse(data, axis=[1])
        data = tf.transpose(data, perm=[1, 0, 2])
        return data

    def _flip_x(self, data):
        data = tf.reverse(data, axis=[0])
        return data

    def _flip_y(self, data):
        data = tf.reverse(data, axis=[1])
        return data

    def _flip_z(self, data):
        data = tf.reverse(data, axis=[2])
        return data

    def _transpose(self, data):
        data = tf.transpose(data)
        return data


    def forward(self, data, state):
        """Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed
            state: Information about the current execution context

        Returns:
            Transformed (augmented) data

        """
        data, mask = data
        key = tf.random.uniform(shape=(), minval=0, maxval=48, dtype=tf.dtypes.int32)
        augment_key = tf.gather_nd(self.isometric_options, [key])
        rotate_y = tf.cast(augment_key[0], tf.bool)
        rotate_z = tf.cast(augment_key[1], tf.bool)
        flip_x = tf.cast(augment_key[2], tf.bool)
        flip_y = tf.cast(augment_key[3], tf.bool)
        flip_z = tf.cast(augment_key[4], tf.bool)
        transpose = tf.cast(augment_key[5], tf.bool)

        if rotate_y:
            data = tf.map_fn(self._rotate_y, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._rotate_y, mask, dtype=tf.dtypes.int32, back_prop=False)

        if rotate_z:
            data = tf.map_fn(self._rotate_z, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._rotate_z, mask, dtype=tf.dtypes.int32, back_prop=False)

        if flip_x:
            data = tf.map_fn(self._flip_x, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._flip_x, mask, dtype=tf.dtypes.int32, back_prop=False)

        if flip_y:
            data = tf.map_fn(self._flip_y, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._flip_y, mask, dtype=tf.dtypes.int32, back_prop=False)

        if flip_z:
            data = tf.map_fn(self._flip_z, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._flip_z, mask, dtype=tf.dtypes.int32, back_prop=False)

        if transpose:
            data = tf.map_fn(self._transpose, data, dtype=tf.dtypes.float32, back_prop=False)
            mask = tf.map_fn(self._transpose, mask, dtype=tf.dtypes.int32, back_prop=False)

        return data, mask
