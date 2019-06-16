import numpy as np


class AbstractAugmentation:
    """
    An abstract class for data augmentation that defines interfaces.
    A custom augmentation can be defined by inheriting from this class.

    Args:
        mode: Augmentation to be applied for training or evaluation.
    """
    def __init__(self, mode="train"):
        self.mode = mode

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
            Transformed data

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
        self.width=None
        self.height=None
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down
        self.transform_matrix = np.eye(3)

    def rotate(self):
        """
        Creates affine transformation matrix for 2D rotation

        Returns:
            Transform affine numpy array
        """
        rotation_range = [0., 0.]
        if type(self.rotation_range) is not tuple and type(self.rotation_range) is not list:
            rotation_range[0] = -1 * self.rotation_range
            rotation_range[1] = self.rotation_range
        else:
            rotation_range = self.rotation_range
        self.rotation_range = rotation_range
        theta = np.random.uniform(low=np.pi / 180 * self.rotation_range[0],
                                  high=np.pi / 180 * self.rotation_range[1])
        base_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        rotation_matrix_1 = np.cos(theta) * base_matrix
        base_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        rotation_matrix_2 = base_matrix * np.sin(theta)
        transform_matrix = rotation_matrix_1 + rotation_matrix_2 + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                                                                               dtype=np.float32)

        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def shift(self):
        """
        Creates affine transformation matrix for 2D shear

        Returns:
            Transform affine numpy array
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
        ty = np.random.uniform(low=self.width_shift_range[0], high=self.width_shift_range[1])
        ty *= self.width
        base_ty = ty * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        tx = np.random.uniform(low=self.height_shift_range[0], high=self.height_shift_range[1])
        tx *= self.height
        base_tx = tx * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
        transform_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   dtype=np.float32) + base_tx + base_ty
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def shear(self):
        """
        Creates affine transformation matrix for 2D zoom / scale

        Returns:
            Transform affine numpy array
        """
        shear_range = [0., 0.]
        if type(self.shear_range) is not tuple and type(self.shear_range) is not list:
            shear_range[0] = -1 * self.shear_range
            shear_range[1] = self.shear_range
        else:
            shear_range = self.shear_range
        self.shear_range = shear_range
        sx = np.random.uniform(low=np.pi / 180 * self.shear_range[0], high=np.pi / 180 * self.shear_range[1])
        sy = np.random.uniform(low=np.pi / 180 * self.shear_range[0], high=np.pi / 180 * self.shear_range[1])
        base_shear1 = -np.sin(sx) * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        base_shear2 = np.cos(sy) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        transform_matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32) + \
                           base_shear1 + base_shear2
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def zoom(self):
        """
        Decides whether or not to flip

        Returns:
            A boolean that represents whether or not to flip
        """
        zoom_range = [0., 0.]
        if type(self.zoom_range) is not tuple and type(self.zoom_range) is not list:
            zoom_range[0] = 1 - self.zoom_range
            zoom_range[1] = 1 + self.zoom_range
        else:
            zoom_range = self.zoom_range
        self.zoom_range = zoom_range
        zx = np.random.uniform(low=self.zoom_range[0], high=self.zoom_range[1])
        zy = np.random.uniform(low=self.zoom_range[0], high=self.zoom_range[1])
        base_zx = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32) / zx
        base_zy = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) / zy
        transform_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32) + base_zx + base_zy
        transform_matrix = self.transform_matrix_offset_center(transform_matrix)
        return transform_matrix

    def flip(self):
        """
        Offsets the tensor to the center of the image

        Args:
            matrix: Affine numpy array

        Returns:
            An affine numpy offset to the center of the image
        """
        do_flip = np.greater(np.random.uniform(low=0, high=1), 0.5)
        return do_flip

    def transform_matrix_offset_center(self, matrix):
        """
        This method set the appropriate variables necessary for the random 2D augmentation. It also computes the
        transformation matrix.

        Returns:
            None
        """
        o_x = self.height / 2 + 0.5
        o_y = self.width / 2 + 0.5
        offset = np.eye(3) + o_x * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]) + o_y * np.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        reset = np.eye(3) + o_x * np.array([[0, 0, -1], [0, 0, 0], [0, 0, 0]]) + o_y * np.array(
            [[0, 0, 0], [0, 0, -1], [0, 0, 0]])
        matrix = np.tensordot(np.tensordot(offset, matrix, axes=1), reset, axes=1)
        return matrix

    def setup(self):
        """
        This method set the appropriate variables necessary for the random 2D augmentation. It also computes the
        transformation matrix.

        Returns:
            None
        """
        transform_matrix = np.identity(3)
        do_flip_lr = False
        do_flip_ud = False
        do_rotate = False
        do_shift = False
        do_zoom = False
        do_shear = False

        if np.isscalar(self.rotation_range):
            if self.rotation_range > 0.:
                do_rotate = True
        else:
            if self.rotation_range[0] > 0. or self.rotation_range[1] > 0.:
                do_rotate = True

        if np.isscalar(self.width_shift_range):
            if self.width_shift_range > 0.:
                do_shift = True
        else:
            if self.width_shift_range[0] > 0. or self.width_shift_range[1] > 0.:
                do_shift = True

        if np.isscalar(self.height_shift_range):
            if self.height_shift_range > 0.:
                do_shift = True
        else:
            if self.height_shift_range[0] > 0. or self.height_shift_range[1] > 0.:
                do_shift = True

        if np.isscalar(self.zoom_range):
            if self.zoom_range != 1.:
                do_zoom = True
        else:
            if self.zoom_range[0] != 1. or self.zoom_range[1] != 0.:
                do_zoom = True

        if np.isscalar(self.shear_range):
            if self.shear_range > 0.:
                do_shear = True
        else:
            if self.shear_range[0] > 0. or self.shear_range[1] > 0.:
                do_shear = True

        if do_rotate:
            if transform_matrix is None:
                transform_matrix = self.rotate()
            else:
                transform_matrix = np.tensordot(transform_matrix, self.rotate(), axes=1)

        if do_shift:
            if transform_matrix is None:
                transform_matrix = self.shift()
            else:
                transform_matrix = np.tensordot(transform_matrix, self.shift(), axes=1)

        if do_zoom:
            if transform_matrix is None:
                transform_matrix = self.zoom()
            else:
                transform_matrix = np.tensordot(transform_matrix, self.zoom(), axes=1)
        if do_shear:
            if transform_matrix is None:
                transform_matrix = self.shear()
            else:
                transform_matrix = np.tensordot(transform_matrix, self.shear(), axes=1)
        if self.flip_left_right:
            do_flip_lr = self.flip()

        if self.flip_up_down:
            do_flip_ud = self.flip()
        self.transform_matrix = transform_matrix
        self.flip_left_right = do_flip_lr
        self.flip_up_down = do_flip_ud

    def transform(self, data):
        """
        Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed

        Returns:
            Transformed (augmented) data
        """
        import cv2

        augment_data = cv2.warpAffine(data, self.transform_matrix[:2, :],
                                                        (data.shape[0], data.shape[1]), flags=cv2.WARP_INVERSE_MAP)
        augment_data = np.fliplr(augment_data) if self.flip_left_right else augment_data
        augment_data = np.flipud(augment_data) if self.flip_up_down else augment_data
        return augment_data



