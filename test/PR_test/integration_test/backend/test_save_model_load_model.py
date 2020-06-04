import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal, one_layer_tf_model, OneLayerTorchModel

def get_model_weight_tf(model):
    weight = []
    for layer in model.layers:
        weight.append(layer.get_weights())

    return weight

def get_model_weight_lenet_torch(model):
    weight = []
    weight.append(model.conv1.weight.data.numpy())
    weight.append(model.conv2.weight.data.numpy())
    weight.append(model.conv3.weight.data.numpy())
    weight.append(model.fc1.weight.data.numpy())
    weight.append(model.fc1.weight.data.numpy())

    return weight


class TestLoadModelAndSaveModel(unittest.TestCase):
    def test_save_model_and_load_model_tf(self):
        m1 = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
        weight1 = get_model_weight_tf(m1)

        fe.backend.save_model(m1, save_dir="tmp", model_name="test")

        m2 = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
        weight2 = get_model_weight_tf(m2)
        self.assertFalse(is_equal(weight1, weight2))

        fe.backend.load_model(m2, weights_path="tmp/test.h5")
        weight3 = get_model_weight_tf(m2)

        self.assertTrue(is_equal(weight1, weight3))

    def test_save_model_and_load_model_torch(self):
        m1 = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
        weight1 = get_model_weight_lenet_torch(m1)

        fe.backend.save_model(m1, save_dir="tmp", model_name="test")

        m2 = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
        weight2 = get_model_weight_lenet_torch(m2)
        self.assertFalse(is_equal(weight1, weight2))

        fe.backend.load_model(m2, weights_path="tmp/test.pt")
        weight3 = get_model_weight_lenet_torch(m2)

        self.assertTrue(is_equal(weight1, weight3))
