import unittest

import numpy as np

from fastestimator.trace.metric import Accuracy
from fastestimator.util import Data


class TestAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([1, 2, 3])
        x_pred = np.array([[1, 1, 3], [2, 3, 4], [1, 1, 0]])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.accuracy = Accuracy(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.accuracy.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of correct'):
            self.assertEqual(self.accuracy.correct, 0)
        with self.subTest('Check initial value of total'):
            self.assertEqual(self.accuracy.total, 0)

    def test_on_batch_end(self):
        self.accuracy.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.accuracy.correct, 1)
        with self.subTest('Check total values'):
            self.assertEqual(self.accuracy.total, 3)

    def test_on_epoch_end(self):
        self.accuracy.correct = 2
        self.accuracy.total = 3
        self.accuracy.on_epoch_end(data=self.data)
        with self.subTest('Check if accuracy value exists'):
            self.assertIn('accuracy', self.data)
        with self.subTest('Check the value of accuracy'):
            self.assertEqual(round(self.data['accuracy'], 2), 0.67)
