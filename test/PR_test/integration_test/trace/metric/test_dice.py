# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

import numpy as np

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.metric import Dice
from fastestimator.util import Data


class TestDice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.array([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                           [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
                           [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]], dtype=np.float32)

        cls.x_pred = np.array([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]],
                                [[1, 0, 1], [1, 0, 1], [0, 1, 0]],
                                [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]], dtype=np.float32)
        cls.dice_output = 0.67777777
        cls.dice = Dice(true_key='x', pred_key='x_pred')
        cls.dice.system = sample_system_object()

    def test_sanity(self):
        data = Data()
        self.dice.on_epoch_begin(data=data)
        self.assertTrue(len(data) == 0)
        data = Data({'x': self.x, 'x_pred': self.x_pred})
        self.dice.on_batch_end(data=data)
        data = Data()
        self.dice.on_epoch_end(data=data)
        with self.subTest('Check if dice exists'):
            self.assertIn('Dice', data)
        with self.subTest('Check the value of dice'):
            self.assertAlmostEqual(data['Dice'], self.dice_output, places=4)
