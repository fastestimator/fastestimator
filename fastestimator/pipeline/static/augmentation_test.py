from unittest import TestCase
from fastestimator.pipeline.static.augmentation import Augmentation
import numpy as np
import tensorflow as tf

eps = 1e-4
rotation_range = [90, 90]
height_shift_range = [0.1, 0.1]
width_shift_range = [0.2, 0.2]
shear_range = [0.2, 0.2]
zoom_range = [2, 2]
width = 5
height = 5
data = np.eye(5)
data = np.expand_dims(data, 2)
data = np.repeat(data, 3, 2)

data_shift = np.array([[1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0.]])
data_shift = np.expand_dims(data_shift, 2)
data_shift = np.repeat(data_shift, 3, 2)


data_rotate = np.array([[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 1., 0., 0.]])
data_rotate = np.expand_dims(data_rotate, 2)
data_rotate = np.repeat(data_rotate, 3, 2)

data_zoom = np.array([[1., 1., 0., 0., 0.],
                       [1., 1., 0., 0., 0.],
                       [0., 0., 1., 1., 0.],
                       [0., 0., 1., 1., 0.],
                       [0., 0., 0., 0., 1.]])
data_zoom = np.expand_dims(data_zoom, 2)
data_zoom = np.repeat(data_zoom, 3, 2)


data_transform = np.array([[0., 0., 1., 1., 0.],
                           [1., 1., 0., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [1., 0., 0., 0., 0.]])
data_transform = np.expand_dims(data_transform, 2)
data_transform = np.repeat(data_transform, 3, 2)

class TestAugmentation(TestCase):
    def test_rotate(self):
        aug = Augmentation(rotation_range=rotation_range)
        transformed_data = self._transform(aug)
        assert np.allclose(transformed_data, data_rotate, rtol=eps)

    def test_shift(self):
        aug = Augmentation(width_shift_range=width_shift_range,
                           height_shift_range=height_shift_range)
        transformed_data = self._transform(aug)
        assert np.allclose(transformed_data, data_shift, rtol=eps)

    def test_shear(self):
        aug = Augmentation(shear_range=shear_range)
        transformed_data = self._transform(aug)
        assert np.allclose(transformed_data, data, rtol=eps)


    def test_zoom(self):
        aug = Augmentation(zoom_range=zoom_range)
        transformed_data = self._transform(aug)
        assert np.allclose(transformed_data, data_zoom, rtol=eps)


    def test_transform(self):
        aug = Augmentation(rotation_range=rotation_range, shear_range=shear_range,
                           width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                           zoom_range=zoom_range)
        transformed_data = self._transform(aug)
        assert np.allclose(transformed_data, data_transform, rtol=eps)

    def _transform(self, aug):
        aug.width = width
        aug.height = height
        aug.setup()
        transformed_data = aug.transform(data)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        return transformed_data
