import unittest

import numpy as np
import torch

from fastestimator.architecture.pytorch import LeNet


class TestLenet(unittest.TestCase):
    def test_lenet(self):
        data = np.ones((1, 1, 28, 28))
        input_data = torch.Tensor(data)
        lenet = LeNet()
        output_shape = lenet(input_data).detach().numpy().shape
        self.assertEqual(output_shape, (1, 10))
