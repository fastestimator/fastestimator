import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestIWD(unittest.TestCase):
    def test_np_input_max_pr95_l5_p1(self):
        l = 5
        pr = 0.95
        p = 1.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr80_l5_p1(self):
        l = 5
        pr = 0.80
        p = 1.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr999_l5_p1(self):
        l = 5
        pr = 0.999
        p = 1.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr95_l16_p3(self):
        l = 16
        pr = 0.95
        p = 3.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr80_l16_p3(self):
        l = 16
        pr = 0.80
        p = 3.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr999_l16_p3(self):
        l = 16
        pr = 0.999
        p = 3.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr95_l64_p025(self):
        l = 64
        pr = 0.95
        p = 0.25
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr80_l64_p025(self):
        l = 64
        pr = 0.80
        p = 0.25
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr999_l64_p025(self):
        l = 64
        pr = 0.999
        p = 0.25
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_max_pr999_l1000_p1(self):
        l = 1000
        pr = 0.999
        p = 1.0
        n = np.array([[0.5] * l, [0] + [1] * (l - 1)])
        target = np.array([[1.0 / l] * l, [pr] + [(1.0 - pr) / (l - 1.0)] * (l - 1)])
        b = fe.backend.iwd(n, power=p, max_prob=pr, pairwise_distance=1.0)
        self.assertTrue(np.allclose(b, target))

    def test_tf_input(self):
        n = tf.convert_to_tensor([[0.5] * 5, [0] + [1] * 4])
        target = tf.convert_to_tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]])
        b = fe.backend.iwd(n)
        self.assertTrue(np.allclose(b, target))

    def test_tf_input_c8(self):
        n = tf.convert_to_tensor([[0.5] * 5, [0] + [8] * 4])
        target = tf.convert_to_tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]])
        b = fe.backend.iwd(n, pairwise_distance=8.0)
        self.assertTrue(np.allclose(b, target))

    def test_torch_input(self):
        n = torch.tensor([[0.5] * 5, [0] + [1] * 4])
        target = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]])
        b = fe.backend.iwd(n)
        self.assertTrue(np.allclose(b, target))

    def test_torch_input_eps(self):
        n = torch.tensor([[0.5] * 5, [0] + [1] * 4])
        eps = torch.tensor(0.062499999999999986)
        target = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.80, 0.05, 0.05, 0.05, 0.05]])
        b = fe.backend.iwd(n, eps=eps)
        self.assertTrue(np.allclose(b, target))
