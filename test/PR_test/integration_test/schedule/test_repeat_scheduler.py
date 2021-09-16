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
import tempfile
import unittest

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDS

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.schedule import RepeatScheduler
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch
from fastestimator.trace import Trace


class TestNonTraceableDataset(TorchDS):
    def __init__(self, data, var):
        super().__init__()
        self.data = data
        self.var = var

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestDataset(NumpyDataset):
    def __init__(self, data, var):
        super().__init__(data)
        self.var = var


class TestNumpyOp(NumpyOp):
    def __init__(self, inputs, outputs, mode, var):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.var = var


class TestTensorOp(TensorOp):
    def __init__(self, inputs, outputs, mode, var):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.var = var


class TestTrace(Trace):
    def __init__(self, var1):
        super().__init__()
        self.var1 = var1


class TestRepeatScheduler(unittest.TestCase):
    def test_save_and_load_state_with_ds_scheduler_tf(self):
        def instantiate_system():
            system = sample_system_object()
            x_train = np.ones((2, 28, 28, 3))
            y_train = np.ones((2, ))

            train_data = RepeatScheduler([
                TestDataset(data={
                    "x": x_train, "y": y_train
                }, var=1),
                TestDataset(data={
                    "x": x_train, "y": y_train
                }, var=1),
            ])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=1)
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.data["train"][None].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        loaded_var = system.pipeline.data["train"][None].get_current_value(1).var
        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_ds_scheduler_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            x_train = np.ones((2, 3, 28, 28))
            y_train = np.ones((2, ))

            train_data = RepeatScheduler([
                TestDataset(data={
                    "x": x_train, "y": y_train
                }, var=1),
                TestDataset(data={
                    "x": x_train, "y": y_train
                }, var=1),
            ])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=1)
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.data["train"][None].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        loaded_var = system.pipeline.data["train"][None].get_current_value(1).var
        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_top_scheduler_tf(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(ops=[
                RepeatScheduler([
                    TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1),
                    TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1)
                ]),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.network.ops[0].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.ops[0].get_current_value(1).var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_top_scheduler_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
            system.network = fe.Network(ops=[
                RepeatScheduler([
                    TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1),
                    TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1)
                ]),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.network.ops[0].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.ops[0].get_current_value(1).var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_nop_scheduler_tf(self):
        def instantiate_system():
            system = sample_system_object()
            system.pipeline.ops = [
                RepeatScheduler([
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1)
                ])
            ]
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.ops[0].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.ops[0].get_current_value(1).var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_nop_scheduler_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            system.pipeline.ops = [
                RepeatScheduler([
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1)
                ])
            ]
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.ops[0].get_current_value(1).var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.ops[0].get_current_value(1).var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_trace_scheduler_tf(self):
        def instantiate_system():
            system = sample_system_object()
            system.traces.append(RepeatScheduler([TestTrace(var1=1), TestTrace(var1=1)]))
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.traces[0].get_current_value(1).var1 = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.traces[0].get_current_value(1).var1

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_trace_scheduler_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            system.traces.append(RepeatScheduler([TestTrace(var1=1), TestTrace(var1=1)]))
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.traces[0].get_current_value(1).var1 = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.traces[0].get_current_value(1).var1

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_non_traceable_entries(self):
        def instantiate_system():
            system = sample_system_object()
            x_train = np.ones((2, 28, 28, 3))
            y_train = np.ones((2, ))
            data = {0: {'x': x_train[0], 'y': y_train[0]}, 1: {'x': x_train[1], 'y': y_train[1]}}
            train_data = RepeatScheduler(
                [TestNonTraceableDataset(data=data, var=3), TestNonTraceableDataset(data=data, var=7)])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=1)
            return system

        system = instantiate_system()

        # make some changes
        new_var1 = 4
        system.pipeline.data["train"][None].get_current_value(1).var = new_var1
        new_var2 = 99
        system.pipeline.data["train"][None].get_current_value(2).var = new_var2

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        with self.subTest('Check that repeat list is still populated'):
            self.assertEqual(2, len(system.pipeline.data["train"][None].repeat_list))
        with self.subTest('Check that classes are still intact'):
            self.assertTrue(isinstance(system.pipeline.data["train"][None].get_current_value(1), TestNonTraceableDataset))
            self.assertTrue(isinstance(system.pipeline.data["train"][None].get_current_value(2), TestNonTraceableDataset))
        with self.subTest('Check that the 1st repeat list entry was not restored'):
            # Since the dataset is not traceable changes shouldn't get restored
            loaded_var = system.pipeline.data["train"][None].get_current_value(1).var
            self.assertEqual(loaded_var, 3)
        with self.subTest('Check that the 2nd repeat list entry was not restored'):
            loaded_var = system.pipeline.data["train"][None].get_current_value(2).var
            self.assertEqual(loaded_var, 7)

    def test_save_and_load_state_with_hybrid_entries(self):
        def instantiate_system():
            system = sample_system_object()
            x_train = np.ones((2, 28, 28, 3))
            y_train = np.ones((2, ))
            data = {0: {'x': x_train[0], 'y': y_train[0]}, 1: {'x': x_train[1], 'y': y_train[1]}}
            train_data = RepeatScheduler([
                TestNonTraceableDataset(data=data, var=3), TestDataset(data={
                    'x': x_train, 'y': y_train
                }, var=7), None
            ])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=1)
            return system

        system = instantiate_system()

        # make some changes
        new_var1 = 4
        system.pipeline.data["train"][None].get_current_value(1).var = new_var1
        new_var2 = 99
        system.pipeline.data["train"][None].get_current_value(2).var = new_var2

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        with self.subTest('Check that repeat list is still populated'):
            self.assertEqual(3, len(system.pipeline.data["train"][None].repeat_list))
        with self.subTest('Check that classes are still intact'):
            self.assertTrue(isinstance(system.pipeline.data["train"][None].get_current_value(1), TestNonTraceableDataset))
            self.assertTrue(isinstance(system.pipeline.data["train"][None].get_current_value(2), TestDataset))
            self.assertTrue(system.pipeline.data["train"][None].get_current_value(3) is None)
        with self.subTest('Check that the 1st repeat list entry was not restored'):
            # Since the dataset is not traceable changes shouldn't get restored
            loaded_var = system.pipeline.data["train"][None].get_current_value(1).var
            self.assertEqual(loaded_var, 3)
        with self.subTest('Check that the 2nd repeat list entry was restored'):
            loaded_var = system.pipeline.data["train"][None].get_current_value(2).var
            self.assertEqual(loaded_var, new_var2)
