import unittest

import numpy as np
import torch

from fastestimator.architecture.pytorch import WideResidualNetwork


class TestLenet(unittest.TestCase):
    def test_wrn(self):
        data = np.ones((1, 3, 32, 32))
        input_data = torch.Tensor(data)
        wrn = WideResidualNetwork(classes=5)
        output_shape = wrn(input_data).detach().numpy().shape
        self.assertEqual(output_shape, (1, 5))

    def test_wrn_depth(self):
        with self.assertRaises(AssertionError):
            wrn = WideResidualNetwork(depth=27)
