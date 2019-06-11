import tensorflow as tf
import math

class AbstractAugmentation:
    """
    An abstract class for data augmentation that defines interfaces.
    A custom augmentation can be defined by inheriting from this class.

    Args:
        mode: Augmentation to be applied for training or evaluation, can be "train", "eval" or "both".
    """
    def __init__(self, mode="train"):
        self.mode = mode
        self.decoded_data = None
        self.feature_name = None

    def setup(self):
        """
        An interface method to be implemented by inheriting augmentation class to setup necessary parameters for the
        augmentation

        Returns:
            None
        """
        return None

    def transform(self, data):
        """
        An interface method to be implemented by inheriting augmentation class to apply the transformation to data

        Args:
            data: Data on which a transformation is to be applied

        Returns:
            Transformed tensor

        """
        return data

class Augmentation(AbstractAugmentation):
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
    def __init__(self, rotation_range=0., width_shift_range=0., height_shift_range=0.,
                shear_range=0., zoom_range=1., flip_left_right=False, flip_up_down=False, mode='train'):
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
        theta = tf.random_uniform([], maxval=math.pi / 180 * self.rotation_range[1],
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
        ty = tf.random_uniform([], maxval=self.width_shift_range[1], minval=self.width_shift_range[0])
        ty *= self.width
        base_ty = ty * tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32)
        tx = tf.random_uniform([], maxval=self.height_shift_range[1], minval=self.height_shift_range[0])
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
        sx = tf.random_uniform([], maxval=math.pi / 180 * self.shear_range[1], minval=math.pi / 180 * self.shear_range[0])
        sy = tf.random_uniform([], maxval=math.pi / 180 * self.shear_range[1], minval=math.pi / 180 * self.shear_range[0])
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
        zx = tf.random_uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        zy = tf.random_uniform([], maxval=self.zoom_range[1], minval=self.zoom_range[0])
        base_zx = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zx
        base_zy = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], shape=[3, 3], dtype=tf.float32) / zy
        transform_matrix = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32) + base_zx + base_zy
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix
        
    def flip(self):
        """
        Decides whether or not to flip

        Returns:
            A boolean that represents whether or not to flip
        """
        do_flip = tf.greater(tf.random_uniform([], minval=0, maxval=1), 0.5)
        return do_flip

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
        transform_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], shape=[3, 3], dtype=tf.float32)
        do_rotate = False
        do_shift = False
        do_zoom = False
        do_shear = False
        self.do_flip_lr_tensor = tf.constant(False)
        self.do_flip_ud_tensor = tf.constant(False)

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
             self.do_flip_lr_tensor = self.flip()
        if self.flip_up_down_boolean:
             self.do_flip_ud_tensor = self.flip()

    def transform(self, data):
        """
        Transforms the data with the augmentation transformation
        
        Args:
            data: Data to be transformed

        Returns:
            Transformed (augmented) data

        """
        transform_matrix_flatten = tf.reshape(self.transform_matrix, shape=[1, 9])
        transform_matrix_flatten = transform_matrix_flatten[0, 0:8]
        augment_data = tf.contrib.image.transform(data, transform_matrix_flatten)
        augment_data = tf.cond(self.do_flip_lr_tensor, lambda: tf.image.flip_left_right(augment_data), lambda: augment_data)
        augment_data = tf.cond(self.do_flip_ud_tensor, lambda: tf.image.flip_up_down(augment_data), lambda: augment_data)
        return augment_data
