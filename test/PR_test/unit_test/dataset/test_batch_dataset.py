import unittest

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.dataset import GeneratorDataset


def inputs():
    while True:
        yield {'x': np.random.rand(16), 'y': np.random.randint(16)}


class TestBatchDataset(unittest.TestCase):
    def test_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        unpaired_ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[2, 2])

        self.assertEqual(len(unpaired_ds), 5)

    def test_split(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})
        train_data.split(0.1)

        self.assertEqual(len(train_data), 54000)
