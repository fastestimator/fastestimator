import unittest

import numpy as np

from fastestimator.test.unittest_util import is_equal
from fastestimator.trace.metric import Recall
from fastestimator.util import Data


class TestRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([1, 2, 3])
        x_pred = np.array([[1, 1, 3], [2, 3, 4], [1, 1, 0]])
        cls.data = Data({'x': x, 'x_pred': x_pred})
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
            self.assertEqual(self.recall.y_true, [1, 2, 3])
        with self.subTest('Check total values'):
            self.assertEqual(self.recall.y_pred, [2, 2, 0])

    def test_on_epoch_end(self):
        self.recall.y_true = [1, 2, 3]
        self.recall.y_pred = [2, 2, 0]
        self.recall.on_epoch_end(data=self.data)
        with self.subTest('Check if recall exists'):
            self.assertIn('recall', self.data)
        with self.subTest('Check the value of recall'):
            self.assertTrue(is_equal(self.data['recall'], np.array([0, 0, 1, 0])))
