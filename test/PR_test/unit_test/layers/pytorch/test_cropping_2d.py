import unittest

import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestCropping2D(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(list(range(100))).view((1, 1, 10, 10))

    def test_cropping_2d_1arg(self):
        op = torch.tensor([[[[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56], [63, 64, 65, 66]]]])
        m = fe.layers.pytorch.Cropping2D(3)
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))

    def test_cropping_2d_2arg(self):
        op = torch.tensor([[[[34, 35], [44, 45], [54, 55], [64, 65]]]])
        m = fe.layers.pytorch.Cropping2D((3, 4))
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))

    def test_cropping_2d_tuple(self):
        op = torch.tensor([[[[14, 15], [24, 25], [34, 35], [44, 45], [54, 55]]]])
        m = fe.layers.pytorch.Cropping2D(((1, 4), 4))
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))
