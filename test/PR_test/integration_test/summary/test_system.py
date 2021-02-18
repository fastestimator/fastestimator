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
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.summary import Summary
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model, sample_system_object, \
    sample_system_object_torch
from fastestimator.trace.trace import Trace


def get_model_names(system):
    model_names = []
    for model in system.network.models:
        model_names.append(model.model_name)
    return model_names


class TestTensorOp(TensorOp):
    def __init__(self, inputs, outputs, mode, var1):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.var1 = var1


class TestTrace(Trace):
    def __init__(self, var1):
        super().__init__()
        self.var1 = var1


def test_model(submodel):
    inp = tf.keras.layers.Input([3])
    x = submodel(inp)
    x = x + 1
    model = tf.keras.models.Model(inputs=inp, outputs=x)
    return model


class TestModel(torch.nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel

    def forward(self, x):
        x = self.submodel(x)
        x = x + 1
        return x


class TestSystem(unittest.TestCase):
    def test_save_and_load_state_torch(self):
        """ `save_state` and `load_state` of an entire system are highly dependent on the implementation of __setstate__
        and __getstate__ of ops, traces, datasets ... etc. The save_state and load_state function should be tested in
        the testing script of that perticular component
        (ex: The test of save_state and load_state with EpochScheduler is located in
             PR_test/integration_test/schedule/test_epoch_scheduler.py)
        """
        system = sample_system_object_torch()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'objects.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.pt')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pt')))

        system = sample_system_object_torch()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(global_step, system.global_step)
            self.assertEqual(epoch_idx, system.epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_save_and_load_state_tf(self):
        """ `save_state` and `load_state` of an entire system are highly dependent on the implementation of __setstate__
        and __getstate__ of ops, traces, datasets ... etc. The save_state and load_state function should be tested in
        the testing script of that particular component
        (ex: The test of save_state and load_state with EpochScheduler is located in
             PR_test/integration_test/schedule/test_epoch_scheduler.py)
        """
        system = sample_system_object()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'objects.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.h5')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pkl')))

        system = sample_system_object()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(global_step, system.global_step)
            self.assertEqual(epoch_idx, system.epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_shared_variables_within_traces(self):
        save_path = tempfile.mkdtemp()

        system = sample_system_object()
        shared_trace_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_trace_var))
        system.traces.append(TestTrace(shared_trace_var))
        shared_trace_var.assign_add(1)

        system.save_state(save_dir=save_path)

        # Re-initialize
        system = sample_system_object()
        shared_trace_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_trace_var))
        system.traces.append(TestTrace(shared_trace_var))

        system.load_state(load_dir=save_path)

        with self.subTest("Check variable value was re-loaded"):
            self.assertEqual(3, system.traces[-1].var1.numpy())
            self.assertEqual(3, system.traces[-2].var1.numpy())

        with self.subTest("Check that variable is still shared"):
            system.traces[-1].var1.assign(5)
            self.assertEqual(5, system.traces[-1].var1.numpy())
            self.assertEqual(5, system.traces[-2].var1.numpy())

        with self.subTest("Check that variable is still linked to outside code"):
            shared_trace_var.assign(7)
            self.assertEqual(7, system.traces[-1].var1.numpy())
            self.assertEqual(7, system.traces[-2].var1.numpy())

    def test_shared_variables_over_object_types(self):
        save_path = tempfile.mkdtemp()

        system = sample_system_object()
        shared_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_var))
        system.network.ops[0].fe_test_var_1 = shared_var
        shared_var.assign_add(1)

        system.save_state(save_dir=save_path)

        # Re-initialize
        system = sample_system_object()
        shared_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_var))
        system.network.ops[0].fe_test_var_1 = shared_var

        system.load_state(load_dir=save_path)

        with self.subTest("Check variable value was re-loaded"):
            self.assertEqual(3, system.traces[-1].var1.numpy())
            self.assertEqual(3, system.network.ops[0].fe_test_var_1.numpy())

        with self.subTest("Check that variable is still shared"):
            system.traces[-1].var1.assign(5)
            self.assertEqual(5, system.traces[-1].var1.numpy())
            self.assertEqual(5, system.network.ops[0].fe_test_var_1.numpy())

        with self.subTest("Check that variable is still linked to outside code"):
            shared_var.assign(7)
            self.assertEqual(7, system.traces[-1].var1.numpy())
            self.assertEqual(7, system.network.ops[0].fe_test_var_1.numpy())

    def test_shared_variable_over_model_tf(self):
        def instantiate_system():
            system = sample_system_object()
            submodel = one_layer_tf_model()
            model = fe.build(model_fn=lambda: test_model(submodel), optimizer_fn='adam', model_name='tf')
            model2 = fe.build(model_fn=lambda: test_model(submodel), optimizer_fn='adam', model_name='tf2')
            system.network = fe.Network(ops=[
                ModelOp(model=model, inputs="x_out", outputs="y_pred"),
                ModelOp(model=model2, inputs="x_out", outputs="y_pred2"),
            ])

            return system

        system = instantiate_system()

        # make some change
        new_weight = [np.array([[1.0], [1.0], [1.0]])]
        system.network.ops[0].model.layers[1].set_weights(new_weight)

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        shared_variable = system.network.ops[0].model.layers[1]
        system.load_state(save_path)

        with self.subTest("Check model varaible was re-loaded"):
            self.assertTrue(is_equal(new_weight, system.network.ops[0].model.layers[1].get_weights()))
            self.assertTrue(is_equal(new_weight, system.network.ops[1].model.layers[1].get_weights()))

        with self.subTest("Check model variable is still shared"):
            new_weight = [np.array([[2.0], [2.0], [2.0]])]
            system.network.ops[0].model.layers[1].set_weights(new_weight)
            self.assertTrue(
                is_equal(system.network.ops[0].model.layers[1].get_weights(),
                         system.network.ops[1].model.layers[1].get_weights()))

        with self.subTest("Check that variable is still linked to outside code"):
            new_weight = [np.array([[3.0], [3.0], [3.0]])]
            shared_variable.set_weights(new_weight)
            self.assertTrue(is_equal(new_weight, system.network.ops[0].model.layers[1].get_weights()))

    def test_shared_variable_over_model_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            submodel = OneLayerTorchModel()
            model = fe.build(model_fn=lambda: TestModel(submodel), optimizer_fn='adam', model_name='torch')
            model2 = fe.build(model_fn=lambda: TestModel(submodel), optimizer_fn='adam', model_name='torch2')
            system.network = fe.Network(ops=[
                ModelOp(model=model, inputs="x_out", outputs="y_pred"),
                ModelOp(model=model2, inputs="x_out", outputs="y_pred2"),
            ])

            return system

        system = instantiate_system()

        # check if multi-gpu system
        if torch.cuda.device_count() > 1:
            model_0 = system.network.ops[0].model.module
            model_1 = system.network.ops[1].model.module
        else:
            model_0 = system.network.ops[0].model
            model_1 = system.network.ops[1].model

        # make some change
        new_weight = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        model_0.submodel.fc1.weight.data = new_weight

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        shared_variable = model_0.submodel.fc1.weight
        system.load_state(save_path)

        with self.subTest("Check model varaible was re-loaded"):
            self.assertTrue(is_equal(new_weight, model_0.submodel.fc1.weight.data))
            self.assertTrue(is_equal(new_weight, model_1.submodel.fc1.weight.data))

        with self.subTest("Check model variable is still shared"):
            new_weight = torch.tensor([[2, 2, 2]], dtype=torch.float32)
            model_0.submodel.fc1.weight.data = new_weight
            self.assertTrue(is_equal(model_0.submodel.fc1.weight.data, model_1.submodel.fc1.weight.data))

        with self.subTest("Check that variable is still linked to outside code"):
            new_weight = torch.tensor([[3, 3, 3]], dtype=torch.float32)
            shared_variable.data = new_weight
            self.assertTrue(is_equal(new_weight, model_0.submodel.fc1.weight.data))

    def test_shared_tf_variable_among_top_trace(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
            var1 = tf.Variable(initial_value=1, trainable=True)
            system.network = fe.Network(ops=[
                TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var1=var1),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            system.traces.append(TestTrace(var1=var1))

            return system

        system = instantiate_system()

        # make some change
        var1_new_val = 2
        system.traces[0].var1.assign(var1_new_val)

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        var1 = system.traces[0].var1
        system.load_state(save_path)

        with self.subTest("Check both trace and tensorop variables are reloaded"):
            self.assertEqual(var1_new_val, system.traces[0].var1.numpy())
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

        with self.subTest("Check trace and tensorop variables are still shared"):
            var1_new_val = 3
            system.traces[0].var1.assign(var1_new_val)
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

        with self.subTest("Check that variable is still linked to outside code"):
            var1_new_val = 4
            var1.assign(var1_new_val)
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

    def test_shared_torch_variable_among_top_trace(self):
        def instantiate_system():
            system = sample_system_object_torch()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
            var1 = torch.tensor(1.0)
            system.network = fe.Network(ops=[
                TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var1=var1),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            system.traces.append(TestTrace(var1=var1))

            return system

        system = instantiate_system()

        # make some change
        var1_new_val = 2.0
        system.traces[0].var1.copy_(torch.tensor(var1_new_val))

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        var1 = system.traces[0].var1
        system.load_state(save_path)

        with self.subTest("Check both trace and tensorop variables are reloaded"):
            self.assertEqual(var1_new_val, system.traces[0].var1.numpy())
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

        with self.subTest("Check trace and tensorop variables are still shared"):
            var1_new_val = 3.0
            system.traces[0].var1.copy_(torch.tensor(var1_new_val))
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

        with self.subTest("Check that variable is still linked to outside code"):
            var1_new_val = 4.0
            var1.copy_(torch.tensor(var1_new_val))
            self.assertEqual(var1_new_val, system.network.ops[0].var1.numpy())

    def test_save_and_load_custom_graphs(self):
        def instantiate_system():
            system = sample_system_object()
            return system

        # Make some data
        example_graph1 = Summary(name="exp1")
        example_graph1.history['train']['acc'][0] = 0.95
        example_graph1.history['train']['acc'][10] = 0.87

        example_graph2 = Summary(name="class1")
        example_graph2.history['eval']['dice'][15] = 0.34
        example_graph2.history['eval']['dice'][25] = 0.75

        example_graph3 = Summary(name="class2")
        example_graph3.history['eval']['dice'][15] = 0.21
        example_graph3.history['eval']['dice'][25] = 0.93

        system = instantiate_system()
        system.add_graph("sample_single_graph", example_graph1)
        system.add_graph("sample_list_graph", [example_graph2, example_graph3])

        # Save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        # Check the results
        with self.subTest('Check that 2 graphs are recovered'):
            self.assertEqual(2, len(system.custom_graphs))
        with self.subTest('Check that the single graph is recovered'):
            self.assertIn("sample_single_graph", system.custom_graphs)
            self.assertEqual('exp1', system.custom_graphs['sample_single_graph'][0].name)
            self.assertEqual(0.87, system.custom_graphs['sample_single_graph'][0].history['train']['acc'][10])
        with self.subTest('Check that multi-graphs are recovered'):
            self.assertIn("sample_list_graph", system.custom_graphs)
            self.assertEqual('class1', system.custom_graphs['sample_list_graph'][0].name)
            self.assertEqual('class2', system.custom_graphs['sample_list_graph'][1].name)
            self.assertEqual(0.34, system.custom_graphs['sample_list_graph'][0].history['eval']['dice'][15])
            self.assertEqual(0.21, system.custom_graphs['sample_list_graph'][1].history['eval']['dice'][15])
