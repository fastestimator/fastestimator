import unittest

import torch

import fastestimator as fe


class TestHadamardCode(unittest.TestCase):
    def test_single_input(self):
        batch_size = 27
        n_features = 123
        n_classes = 10
        x = torch.ones((batch_size, n_features))
        layer = fe.layers.pytorch.HadamardCode(in_features=n_features, n_classes=n_classes)
        y = layer(x)
        self.assertEqual((batch_size, n_classes), y.shape)
        for row in y:
            # Should have produced a probability vector, so each row must be equal to 1
            self.assertAlmostEqual(1, row.sum().detach().numpy(), delta=0.000001)

    def test_multi_input(self):
        batch_size = 32
        n_features = 18
        n_classes = 18
        x = [torch.ones((batch_size, n_features)) for _ in range(6)]
        layer = fe.layers.pytorch.HadamardCode(in_features=[n_features] * len(x), n_classes=n_classes)
        y = layer(x)
        self.assertEqual((batch_size, n_classes), y.shape)
        for row in y:
            # Should have produced a probability vector, so each row must be equal to 1
            self.assertAlmostEqual(1, row.sum().detach().numpy(), delta=0.000001)

    def test_longer_code(self):
        batch_size = 16
        n_features = 11
        n_classes = 13
        x = torch.ones((batch_size, n_features))
        layer = fe.layers.pytorch.HadamardCode(in_features=n_features, n_classes=n_classes, code_length=256)
        y = layer(x)
        self.assertEqual((batch_size, n_classes), y.shape)
        for row in y:
            # Should have produced a probability vector, so each row must be equal to 1
            self.assertAlmostEqual(1, row.sum().detach().numpy(), delta=0.000001)

    def test_shorter_code(self):
        n_classes = 10
        self.assertRaises(ValueError,
                          lambda: fe.layers.pytorch.HadamardCode(in_features=22, n_classes=n_classes, code_length=8))

    def test_non_power_of_two_code_length(self):
        n_classes = 10
        self.assertRaises(ValueError,
                          lambda: fe.layers.pytorch.HadamardCode(in_features=22, n_classes=n_classes, code_length=18))

    def test_negative_code_length(self):
        self.assertRaises(ValueError,
                          lambda: fe.layers.pytorch.HadamardCode(in_features=22, n_classes=5, code_length=-2))
