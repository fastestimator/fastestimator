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

import tensorflow as tf

import tensorflow_probability as tfp
from fastestimator.util.op import TensorOp


class Augmentation2D(TensorOp):
    """
   This class supports commonly used 2D random affine transformations for data augmentation.
   Either a scalar ``x`` or a tuple ``[x1, x2]`` can be specified for rotation, shearing, shifting, and zoom.

   Args:
       rotation_range: Scalar (x) that represents the range of random rotation (in degrees) from -x to x /
        Tuple ([x1, x2]) that represents  the range of random rotation between x1 and x2.
       width_shift_range: Float (x) that represents the range of random width shift (in pixels) from -x to x /
        Tuple ([x1, x2]) that represents  the range of random width shift between x1 and x2.
       height_shift_range: Float (x) that represents the range of random height shift (in pixels) from -x to x /
        Tuple ([x1, x2]) that represents  the range of random height shift between x1 and x2.
       shear_range: Scalar (x) that represents the range of random shear (in degrees) from -x to x /
        Tuple ([x1, x2]) that represents  the range of random shear between x1 and x2.
       zoom_range: Float (x) that represents the range of random zoom (in percentage) from -x to x /
        Tuple ([x1, x2]) that represents  the range of random zoom between x1 and x2.
       flip_left_right: Boolean representing whether to flip the image horizontally with a probability of 0.5.
       flip_up_down: Boolean representing whether to flip the image vertically with a probability of 0.5.
       mode: Augmentation on 'training' data or 'evaluation' data.
   """
    def __init__(self, rotation_range=0., width_shift_range=0., height_shift_range=0., shear_range=0., zoom_range=1.,
                 flip_left_right=False, flip_up_down=False, mode='train'):
        self.mode = mode
        self.height = None
        self.width = None
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.flip_left_right_boolean = flip_left_right
        self.flip_up_down_boolean = flip_up_down
        self.transform_matrix = tf.eye(3)
        # There appears to be a bug in TF2 which breaks compatibility between boolean control flow and gradient tapes.
        # Using 0/1 instead for the time being. Test against future versions of TF by running caricature visualization
        self.do_flip_lr_tensor = tf.Variable(0, trainable=False)
        self.do_flip_ud_tensor = tf.Variable(0, trainable=False)

    def rotate(self):
        """
        Creates affine transformation matrix for 2D rotation

        Returns:
            Transform affine tensor
        """
        rotation_range = [0., 0.]
        if type(self.rotation_range) is not tuple and type(self.rotation_range) is not list:
            rotation_range[0] = -1 * self.rotation_range
            rotation_range[1] = self.rotation_range
        else:
            rotation_range = self.rotation_range
        self.rotation_range = rotation_range
        theta = tf.random.uniform([], maxval=math.pi / 180 * self.rotation_range[1],
                                  minval=math.pi / 180 * self.rotation_range[0])
        base_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        rotation_matrix_1 = tf.cos(theta) * base_matrix
        base_matrix = tf.constant([[0, -1, 0], [1, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        rotation_matrix_2 = base_matrix * tf.sin(theta)
        transform_matrix = rotation_matrix_1 + rotation_matrix_2 + tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                                                                               shape=[3, 3], dtype=tf.float32)
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def shift(self):
        """
        Creates affine transformation matrix for 2D shift

        Returns:
            Transform affine tensor
        """
        width_shift_range = [0., 0.]
        if type(self.rotation_range) is not tuple and type(self.width_shift_range) is not list:
            width_shift_range[0] = -1 * self.width_shift_range
            width_shift_range[1] = self.width_shift_range
        else:
            width_shift_range = self.width_shift_range
        self.width_shift_range = width_shift_range
        height_shift_range = [0., 0.]
        if type(self.height_shift_range) is not tuple and type(self.height_shift_range) is not list:
            height_shift_range[0] = -1 * self.height_shift_range
            height_shift_range[1] = self.height_shift_range
        else:
            height_shift_range = self.height_shift_range
        self.height_shift_range = height_shift_range
        ty = tf.random.uniform([], maxval=self.width_shift_range[1], minval=self.width_shift_range[0])
        ty *= self.width
        base_ty = ty * tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        tx = tf.random.uniform([], maxval=self.height_shift_range[1], minval=self.height_shift_range[0])
        tx *= self.height
        base_tx = tx * tf.constant([[0, 0, 0], [0, 0, 1], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        transform_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], shape=[3, 3],
                                       dtype=tf.float32) + base_tx + base_ty
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def shear(self):
        """
        Creates affine transformation matrix for 2D shear

        Returns:
            Transform affine tensor
        """
        shear_range = [0., 0.]
        if type(self.shear_range) is not tuple and type(self.shear_range) is not list:
            shear_range[0] = -1 * self.shear_range
            shear_range[1] = self.shear_range
        else:
            shear_range = self.shear_range
        self.shear_range = shear_range
        sx = tf.random.uniform([], maxval=math.pi / 180 * self.shear_range[1],
                               minval=math.pi / 180 * self.shear_range[0])
        sy = tf.random.uniform([], maxval=math.pi / 180 * self.shear_range[1],
                               minval=math.pi / 180 * self.shear_range[0])
        base_shear1 = -tf.sin(sx) * tf.constant([[0, 1, 0], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        base_shear2 = tf.cos(sy) * tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        transform_matrix = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32) + \
                           base_shear1 + base_shear2
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def zoom(self):
        """
        Creates affine transformation matrix for 2D zoom / scale

        Returns:
            Transform affine tensor
        """
        zoom_range = [0., 0.]
        if type(self.zoom_range) is not tuple and type(self.zoom_range) is not list:
            zoom_range[0] = 1 - self.zoom_range
            zoom_range[1] = 1 + self.zoom_range
        else:
            zoom_range = self.zoom_range
        self.zoom_range = zoom_range
        zx = tf.random.uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        zy = tf.random.uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        base_zx = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zx
        base_zy = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zy
        transform_matrix = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], shape=[3, 3],
                                       dtype=tf.float32) + base_zx + base_zy
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def flip(self):
        """
        Decides whether or not to flip

        Returns:
            A boolean that represents whether or not to flip
        """
        do_flip = tf.greater(tf.random.uniform([], minval=0, maxval=1), 0.5)
        return 1 if do_flip else 0

    def transform_matrix_offset_center(self, matrix):
        """
        Offsets the tensor to the center of the image

        Args:
            matrix: Affine tensor

        Returns:
            An affine tensor offset to the center of the image
        """
        o_x = self.height / tf.constant([2], dtype=tf.float32) + tf.constant([0.5], dtype=tf.float32)
        o_y = self.width / tf.constant([2], dtype=tf.float32) + tf.constant([0.5], dtype=tf.float32)
        eye = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32)

        offset_matrix = eye + \
                        tf.multiply(o_x,
                                    tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)) + \
                        tf.multiply(o_y, tf.constant([[0, 0, 0], [0, 0, 1], [0, 0, 0]], shape=[3, 3], dtype=tf.float32))

        reset_matrix = eye + \
                       tf.multiply(o_x,
                                   tf.constant([[0, 0, -1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)) + \
                       tf.multiply(tf.constant([[0, 0, 0], [0, 0, -1], [0, 0, 0]], shape=[3, 3], dtype=tf.float32), o_y)

        transform_matrix = tf.tensordot(tf.tensordot(offset_matrix, matrix, axes=1), reset_matrix, axes=1)
        return transform_matrix

    def setup(self):
        """
        This method set the appropriate variables necessary for the random 2D augmentation. It also computes the
        transformation matrix.

        Returns:
            None
        """
        # \NOTE(JP): tracing behavior from dataset.map causes issue when any tensor id defined as tf.constant
        transform_matrix = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        do_rotate = False
        do_shift = False
        do_zoom = False
        do_shear = False
        self.do_flip_lr_tensor.assign(0)
        self.do_flip_ud_tensor.assign(0)

        if type(self.rotation_range) is not tuple and type(self.rotation_range) is not list:
            if self.rotation_range > 0.:
                do_rotate = True
        else:
            if self.rotation_range[0] > 0. or self.rotation_range[1] > 0.:
                do_rotate = True

        if type(self.width_shift_range) is not tuple and type(self.width_shift_range) is not list:
            if self.width_shift_range > 0.:
                do_shift = True
        else:
            if self.width_shift_range[0] > 0. or self.width_shift_range[1] > 0.:
                do_shift = True

        if type(self.height_shift_range) is not tuple and type(self.height_shift_range) is not list:
            if self.height_shift_range > 0.:
                do_shift = True
        else:
            if self.height_shift_range[0] > 0. or self.height_shift_range[1] > 0.:
                do_shift = True

        if type(self.zoom_range) is not tuple and type(self.zoom_range) is not list:
            if self.zoom_range != 1.:
                do_zoom = True
        else:
            if self.zoom_range[0] != 1. or self.zoom_range[1] != 0.:
                do_zoom = True

        if type(self.shear_range) is not tuple and type(self.shear_range) is not list:
            if self.shear_range > 0.:
                do_shear = True
        else:
            if self.shear_range[0] > 0. or self.shear_range[1] > 0.:
                do_shear = True

        if do_rotate:
            if transform_matrix is None:
                transform_matrix = self.rotate()
            else:
                transform_matrix = tf.tensordot(transform_matrix, self.rotate(), axes=1)

        if do_shift:
            if transform_matrix is None:
                transform_matrix = self.shift()
            else:
                transform_matrix = tf.tensordot(transform_matrix, self.shift(), axes=1)

        if do_zoom:
            if transform_matrix is None:
                transform_matrix = self.zoom()
            else:
                transform_matrix = tf.tensordot(transform_matrix, self.zoom(), axes=1)
        if do_shear:
            if transform_matrix is None:
                transform_matrix = self.shear()
            else:
                transform_matrix = tf.tensordot(transform_matrix, self.shear(), axes=1)

        self.transform_matrix = transform_matrix

        if self.flip_left_right_boolean:
            self.do_flip_lr_tensor.assign(self.flip())
        if self.flip_up_down_boolean:
            self.do_flip_ud_tensor.assign(self.flip())

    def forward(self, data, state):
        """
        Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed
            state: Information about the current execution context

        Returns:
            Transformed (augmented) data

        """
        augment_data = self._transform(data)

        if self.do_flip_lr_tensor == 1:
            augment_data = tf.image.flip_left_right(augment_data)
        if self.do_flip_ud_tensor == 1:
            augment_data = tf.image.flip_up_down(augment_data)

        return augment_data

    def _transform(self, data):
        dtype = data.dtype
        x_shape, y_shape, z_shape = data.get_shape()[0], data.get_shape()[1], data.get_shape()[2]
        x_range = tf.range(x_shape)
        y_range = tf.range(y_shape)
        z_range = tf.range(z_shape)
        x_, y_, z_ = tf.meshgrid(x_range, y_range, z_range, indexing='ij')
        x_ = tf.reshape(x_, [-1])
        y_ = tf.reshape(y_, [-1])
        z_ = tf.reshape(z_, [-1])
        coords = tf.stack([x_, y_, tf.ones_like(x_)])

        M = tf.linalg.inv(self.transform_matrix)
        coords = tf.matmul(tf.cast(M, tf.float32), tf.cast(coords, tf.float32))

        x_ = tf.cast(coords[0], tf.int32)
        y_ = tf.cast(coords[1], tf.int32)

        mask = (x_ > -1) & (x_ < x_shape) & (y_ > -1) & (y_ < y_shape)
        # mask_inv = ~mask
        mask = tf.cast(mask, dtype)
        # mask_inv = tf.cast(mask_inv, tf.int32)
        # mask_inv = tf.multiply(mask_inv, fill_val)
        x_ = tf.cast(tf.clip_by_value(tf.round(x_), 0, x_shape - 1), tf.int32)

        y_ = tf.cast(tf.clip_by_value(tf.round(y_), 0, y_shape - 1), tf.int32)

        result_flat = tf.gather_nd(data, tf.stack([x_, y_, z_], axis=-1))
        # result_flat = result_flat * tf.cast(mask, tf.int32) +
        result_flat = tf.multiply(result_flat, mask)
        return tf.convert_to_tensor(tf.reshape(result_flat, data.shape))


class MixUpBatch(TensorOp):
    """
    This class should be used in conjunction with MixUpLoss to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412)
    """
    def __init__(self, inputs=None, outputs=None, mode=None, alpha=1.0):
        """
        Args:
            inputs: key of the input to be mixed up
            outputs: key to store the mixed-up input
            mode: what mode to execute in. Probably 'train'
            alpha: the alpha value defining the beta distribution to be drawn from during training
            warmup: how many steps (int) to wait before starting mix-up
        """
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = tf.constant(alpha)
        self.beta = tfp.distributions.Beta(alpha, alpha)

    def forward(self, data, state):
        iterdata = data if isinstance(data, list) else list(data) if isinstance(data, tuple) else [data]
        lam = self.beta.sample()
        # Could do random mix-up using tf.gather() on a shuffled index list, but batches are already randomly ordered,
        # so just need to roll by 1 to get a random combination of inputs. This also allows MixUpLoss to easily compute
        # the corresponding Y values
        mix = [lam * dat + (1.0 - lam) * tf.roll(dat, shift=1, axis=0) for dat in iterdata]
        return mix + [lam]


class AdversarialSample(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None, epsilon=0.1):
        assert len(inputs) == 2, \
            "AdversarialSample requires 2 inputs: a loss value and the input data which lead to the loss"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.epsilon = epsilon

    def forward(self, data, state):
        clean_loss, clean_data = data
        tape = state['tape']
        with tape.stop_recording():
            grad_clean = tape.gradient(clean_loss, clean_data)
            adverse_data = tf.clip_by_value(clean_data + self.epsilon * tf.sign(grad_clean), tf.reduce_min(clean_data),
                                            tf.reduce_max(clean_data))
        return adverse_data


class Average(TensorOp):
    def forward(self, data, state):
        iterdata = data if isinstance(data, list) else list(data) if isinstance(data, tuple) else [data]
        num_entries = len(iterdata)
        result = 0.0
        for elem in iterdata:
            result += elem
        return result / num_entries
