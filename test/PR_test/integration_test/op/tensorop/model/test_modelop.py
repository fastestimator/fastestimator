import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model


class TestModelOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = {'mode': 'train', 'epoch': 1}
        cls.tf_input_data = tf.constant([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        cls.torch_input_data = torch.Tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        cls.output = np.array([[6.0], [-2.5]])

    def test_tf_input(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        op = ModelOp(inputs='x', outputs='x', model=model)
        output = op.forward(data=self.tf_input_data, state=self.state)
        self.assertTrue(is_equal(output.numpy(), self.output))

    def test_torch_input(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        op = ModelOp(inputs='x', outputs='x', model=model)
        output = op.forward(data=self.torch_input_data, state=self.state)
        self.assertTrue(is_equal(output.detach().numpy(), self.output))

