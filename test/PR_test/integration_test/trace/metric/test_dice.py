import unittest

import numpy as np

from fastestimator.trace.metric import Dice
from fastestimator.util import Data


class TestDice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([1, 2, 3])
        x_pred = np.array([[1, 1, 3], [2, 3, 4], [1, 1, 0]])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.dice_output = [1.4999999987500001, 2.3999999972, 2.3999999972]
        cls.dice = Dice(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.dice.on_epoch_begin(data=self.data)
        self.assertEqual(self.dice.dice, [])

    def test_on_batch_end(self):
        self.dice.dice = []
        self.dice.on_batch_end(data=self.data)
        self.assertEqual(self.dice.dice, self.dice_output)

    def test_on_epoch_end(self):
        self.dice.dice = [1.4999999987500001, 2.3999999972, 2.3999999972]
        self.dice.on_epoch_end(data=self.data)
        with self.subTest('Check if dice exists'):
            self.assertIn('Dice', self.data)
        with self.subTest('Check the value of dice'):
            self.assertEqual(self.data['Dice'], 2.0999999977166666)
