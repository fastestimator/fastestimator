import unittest

import numpy as np

from fastestimator.test.unittest_util import is_equal
from fastestimator.trace.metric import Recall
from fastestimator.util import Data


class TestRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        x_binary = np.array([1])
        x_pred_binary = np.array([0.9])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.data_binary = Data({'x': x_binary, 'x_pred': x_pred_binary})
        cls.recall = Recall(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.recall.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.recall.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.recall.y_pred, [])

    def test_on_batch_end(self):
        self.recall.y_true = []
        self.recall.y_pred = []
        self.recall.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.recall.y_true, [1, 1])
        with self.subTest('Check total values'):
            self.assertEqual(self.recall.y_pred, [1, 0])

    def test_on_epoch_end(self):
        self.recall.y_true = [1, 1]
        self.recall.y_pred = [1, 0]
        self.recall.binary_classification = False
        self.recall.on_epoch_end(data=self.data)
        with self.subTest('Check if recall exists'):
            self.assertIn('recall', self.data)
        with self.subTest('Check the value of recall'):
            self.assertTrue(is_equal(self.data['recall'], np.array([0, 0.5])))

    def test_on_batch_end_binary_classification(self):
        self.recall.y_true = []
        self.recall.y_pred = []
        self.recall.on_batch_end(data=self.data_binary)
        with self.subTest('Check correct values'):
            self.assertEqual(self.recall.y_true, [1])
        with self.subTest('Check total values'):
            self.assertEqual(self.recall.y_pred, [1.0])

    def test_on_epoch_end_binary_classification(self):
        self.recall.y_true = [1]
        self.recall.y_pred = [1.0]
        self.recall.binary_classification = True
        self.recall.on_epoch_end(data=self.data_binary)
        with self.subTest('Check if recall exists'):
            self.assertIn('recall', self.data_binary)
        with self.subTest('Check the value of recall'):
            self.assertEqual(self.data_binary['recall'], 1.0)
