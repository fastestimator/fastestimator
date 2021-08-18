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
import os
import tempfile
import unittest

import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import MultiLayerTorchModel, is_equal, one_layer_tf_model, sample_system_object
from fastestimator.trace.io import ModelSaver
from fastestimator.util.data import Data


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
        cls.save_dir = tempfile.mkdtemp()

    def test_tf_model(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=self.save_dir)
        model_saver.system = sample_system_object()
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path = os.path.join(self.save_dir, model_name + '.h5')
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
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        torch_model_path = os.path.join(self.save_dir, model_name + '.pt')
        if os.path.exists(torch_model_path):
            os.remove(torch_model_path)
        model_saver.on_epoch_end(data=Data())
        with self.subTest('Check if model is saved'):
            self.assertTrue(os.path.exists(torch_model_path))
        with self.subTest('Validate model weights'):
            m2 = fe.build(model_fn=MultiLayerTorchModelWithoutWeights, optimizer_fn='adam')
            fe.backend.load_model(m2, torch_model_path)
            self.assertTrue(is_equal(list(m2.parameters()), list(model.parameters())))

    def test_max_to_keep_tf(self):
        save_dir = tempfile.mkdtemp()
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=save_dir, max_to_keep=2)
        model_saver.system = sample_system_object()
        model_saver.on_epoch_end(data=Data())
        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path1 = os.path.join(save_dir, model_name + '.h5')

        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path2 = os.path.join(save_dir, model_name + '.h5')

        with self.subTest('Check only two files are kept'):
            self.assertEqual(len(os.listdir(save_dir)), 2)

        with self.subTest('Check two latest models are kept'):
            self.assertTrue(os.path.exists(tf_model_path1))
            self.assertTrue(os.path.exists(tf_model_path2))

    def test_max_to_keep_tf_architecture(self):
        save_dir = tempfile.mkdtemp()
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=save_dir, max_to_keep=2, save_architecture=True)
        model_saver.system = sample_system_object()
        model_saver.on_epoch_end(data=Data())
        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path1 = os.path.join(save_dir, model_name + '.h5')
        tf_architecture_path1 = os.path.join(save_dir, model_name)

        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        tf_model_path2 = os.path.join(save_dir, model_name + '.h5')
        tf_architecture_path2 = os.path.join(save_dir, model_name)

        with self.subTest('Check only four files are kept'):
            self.assertEqual(len(os.listdir(save_dir)), 4)

        with self.subTest('Check two latest models are kept'):
            self.assertTrue(os.path.exists(tf_model_path1))
            self.assertTrue(os.path.exists(tf_model_path2))
            self.assertTrue(os.path.exists(tf_architecture_path1))
            self.assertTrue(os.path.isdir(tf_architecture_path1))
            self.assertTrue(os.path.exists(tf_architecture_path2))
            self.assertTrue(os.path.isdir(tf_architecture_path2))

    def test_max_to_keep_torch(self):
        save_dir = tempfile.mkdtemp()
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        model_saver = ModelSaver(model=model, save_dir=save_dir, max_to_keep=2)
        model_saver.system = sample_system_object()
        model_saver.on_epoch_end(data=Data())
        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        torch_model_path1 = os.path.join(save_dir, model_name + '.pt')

        model_saver.system.epoch_idx += 1
        model_saver.on_epoch_end(data=Data())
        model_name = "{}_epoch_{}".format(model_saver.model.model_name, model_saver.system.epoch_idx)
        torch_model_path2 = os.path.join(save_dir, model_name + '.pt')

        with self.subTest('Check only two file are kept'):
            self.assertEqual(len(os.listdir(save_dir)), 2)

        with self.subTest('Check two latest model are kept'):
            self.assertTrue(os.path.exists(torch_model_path1))
            self.assertTrue(os.path.exists(torch_model_path2))

    def test_torch_architecture_save(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        save_dir = tempfile.mkdtemp()
        with self.assertRaises(ValueError):
            ModelSaver(model=model, save_dir=save_dir, save_architecture=True)

    def test_max_to_keep_invalid_value(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        save_dir = tempfile.mkdtemp()

        with self.subTest('Check max_to_keep < 0'):
            with self.assertRaises(ValueError):
                ModelSaver(model=model, save_dir=save_dir, max_to_keep=-2)
