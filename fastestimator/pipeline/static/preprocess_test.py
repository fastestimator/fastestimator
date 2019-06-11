from unittest import TestCase
from fastestimator.pipeline.static.preprocess import Zscore, Minmax, Scale, Onehot, Resize, Reshape, Binarize
import tensorflow as tf
import numpy as np

eps = 1e-4
shape = (5, 5, 3)
data = 255 * np.eye(5)
data = np.expand_dims(data, 2)
img = np.repeat(data, 3, 2)

labels = np.array([2])
scalar = 0.2
num_dims = 3
resize_shape = np.array([3, 3])
reshape_shape = (shape[1], shape[0] * shape[2])

img_zscore = np.array([[ 2. , -0.5, -0.5, -0.5, -0.5],
                       [-0.5,  2. , -0.5, -0.5, -0.5],
                       [-0.5, -0.5,  2. , -0.5, -0.5],
                       [-0.5, -0.5, -0.5,  2. , -0.5],
                       [-0.5, -0.5, -0.5, -0.5,  2. ]], dtype=np.float32)
img_zscore = np.expand_dims(img_zscore, 2)
img_zscore = np.repeat(img_zscore, 3, 2)

img_minmax = np.eye(5, dtype=np.float32)
img_minmax = np.expand_dims(img_minmax, 2)
img_minmax = np.repeat(img_minmax, 3, 2)

img_scale = 51 * np.eye(5, dtype=np.float32)
img_scale = np.expand_dims(img_scale, 2)
img_scale = np.repeat(img_scale, 3, 2)

labels_onehot = np.array([[0., 0., 1.]])

img_resize = np.array([[255., 0., 0.],
                       [0., 141.66666, 0.],
                       [0.,   0., 141.66669]], dtype=np.float32)
img_resize = np.expand_dims(img_resize, 2)
img_resize = np.repeat(img_resize, 3, 2)

img_reshape = np.array([[255., 255., 255.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0., 255., 255., 255.,   0.,   0.,   0.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0., 255., 255., 255.,   0.,   0.,
                          0.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 255., 255.,
                        255.,   0.,   0.,   0.],
                       [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                          0., 255., 255., 255.]])


class TestZscore(TestCase):
    def test_transform(self):
        preprocess = Zscore()
        transformed_data = preprocess.transform(img)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, img_zscore, rtol=eps)


class TestMinmax(TestCase):
    def test_transform(self):
        preprocess = Minmax()
        transformed_data = preprocess.transform(img)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, img_minmax, rtol=eps)


class TestScale(TestCase):
    def test_transform(self):
        preprocess = Scale(scalar)
        transformed_data = preprocess.transform(img)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, img_scale, rtol=eps)


class TestOnehot(TestCase):
    def test_transform(self):
        preprocess = Onehot(num_dims)
        transformed_data = preprocess.transform(labels)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, labels_onehot, rtol=eps)


class TestReshape(TestCase):
    def test_transform(self):
        preprocess = Reshape(reshape_shape)
        transformed_data = preprocess.transform(img)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, img_reshape, rtol=eps)


class TestBinarize(TestCase):
    def test_transform(self):
        preprocess = Binarize(0.5)
        preprocess.transform(img)


class TestResize(TestCase):
    def test_transform(self):
        preprocess = Resize(resize_shape)
        transformed_data = preprocess.transform(img)
        sess = tf.Session()
        transformed_data = sess.run(transformed_data)
        assert np.allclose(transformed_data, img_resize, rtol=eps)
