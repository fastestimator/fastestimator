import unittest

import numpy as np

from fastestimator.test.unittest_util import is_equal
from fastestimator.trace.metric import Precision
from fastestimator.util import Data


class TestPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        x_binary = np.array([1])
        x_pred_binary = np.array([0.9])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.data_binary = Data({'x': x_binary, 'x_pred': x_pred_binary})
        cls.precision = Precision(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.precision.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.precision.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.precision.y_pred, [])

    def test_on_batch_end(self):
        self.precision.y_true = []
        self.precision.y_pred = []
        self.precision.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.precision.y_true, [1, 1])
        with self.subTest('Check total values'):
            self.assertEqual(self.precision.y_pred, [1, 0])

    def test_on_epoch_end(self):
        self.precision.y_true = [1, 1]
        self.precision.y_pred = [1, 0]
        self.precision.binary_classification = False
        self.precision.on_epoch_end(data=self.data)
        with self.subTest('Check if precision exists'):
            self.assertIn('precision', self.data)
        with self.subTest('Check the value of precision'):
            self.assertTrue(is_equal(self.data['precision'], np.array([0, 1])))

    def test_on_batch_end_binary_classification(self):
        self.precision.y_true = []
        self.precision.y_pred = []
        self.precision.on_batch_end(data=self.data_binary)
        with self.subTest('Check correct values'):
            self.assertEqual(self.precision.y_true, [1])
        with self.subTest('Check total values'):
            self.assertEqual(self.precision.y_pred, [1.0])

    def test_on_epoch_end_binary_classification(self):
        self.precision.y_true = [1]
        self.precision.y_pred = [1.0]
        self.precision.binary_classification = True
        self.precision.on_epoch_end(data=self.data_binary)
        with self.subTest('Check if precision exists'):
            self.assertIn('precision', self.data_binary)
        with self.subTest('Check the value of precision'):
            self.assertEqual(self.data_binary['precision'], 1.0)
