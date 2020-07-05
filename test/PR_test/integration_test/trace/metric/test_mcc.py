import unittest

import numpy as np

from fastestimator.trace.metric import MCC
from fastestimator.util import Data


class TestMCC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([1, 2, 3])
        x_pred = np.array([[1, 1, 3], [2, 3, 4], [1, 1, 0]])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.mcc = MCC(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.mcc.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.mcc.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.mcc.y_pred, [])

    def test_on_batch_end(self):
        self.mcc.y_true = []
        self.mcc.y_pred = []
        self.mcc.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.mcc.y_true, [1, 2, 3])
        with self.subTest('Check total values'):
            self.assertEqual(self.mcc.y_pred, [2, 2, 0])

    def test_on_epoch_end(self):
        self.mcc.y_true = [1, 2, 3]
        self.mcc.y_pred = [2, 2, 0]
        self.mcc.on_epoch_end(data=self.data)
        with self.subTest('Check if mcc exists'):
            self.assertIn('mcc', self.data)
        with self.subTest('Check the value of mcc'):
            self.assertEqual(self.data['mcc'], 0.20412414523193154)
