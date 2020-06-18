import unittest

import tensorflow as tf

import fastestimator as fe


class TestNumpyDataset(unittest.TestCase):
    def test_dataset(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})

        self.assertEqual(len(train_data), 60000)
