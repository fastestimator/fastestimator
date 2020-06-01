import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestGetLr(unittest.TestCase):
    def test_get_lr_tf(self):
        m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 1e-4))

        m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn=lambda: tf.optimizers.Adam(5e-2))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 5e-2))


    def test_get_lr_torch(self):
        m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1e-4))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 1e-4))

        m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=5e-2))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 5e-2))
