import os
import tempfile
import unittest

import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import MultiLayerTorchModel, is_equal, one_layer_tf_model, sample_system_object
from fastestimator.trace.io import ModelSaver


def one_layer_model_without_weights():
    input = tf.keras.layers.Input([3])
    x = tf.keras.layers.Dense(units=1, use_bias=False)(input)
    model = tf.keras.models.Model(inputs=input, outputs=x)
    return model


class MultiLayerTorchModelWithoutWeights(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 2, bias=False)
        self.fc2 = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestModelSaver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_dir = tempfile.gettempdir()

    def test_tf_model(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=self.save_dir)
        model_saver.system = sample_system_object()
        model_saver.on_epoch_end(data={})
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path = os.path.join(self.save_dir, model_name+'.h5')
        with self.subTest('Check if model is saved'):
            self.assertTrue(os.path.exists(tf_model_path))
        with self.subTest('Validate model weights'):
            m2 = fe.build(model_fn=one_layer_model_without_weights, optimizer_fn='adam')
            fe.backend.load_model(m2, tf_model_path)
            self.assertTrue(is_equal(m2.trainable_variables, model.trainable_variables))

    def test_torch_model(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=self.save_dir)
        model_saver.system = sample_system_object()
        model_name = model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        torch_model_path = os.path.join(self.save_dir, model_name+'.pt')
        if os.path.exists(torch_model_path):
            os.remove(torch_model_path)
        model_saver.on_epoch_end(data={})
        with self.subTest('Check if model is saved'):
            self.assertTrue(os.path.exists(torch_model_path))
        with self.subTest('Validate model weights'):
            m2 = fe.build(model_fn=MultiLayerTorchModelWithoutWeights, optimizer_fn='adam')
            fe.backend.load_model(m2, torch_model_path)
            self.assertTrue(is_equal(list(m2.parameters()), list(model.parameters())))
