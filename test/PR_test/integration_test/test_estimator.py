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
import unittest
from io import StringIO

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.autograph.impl.api import StagingError
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe
from fastestimator.architecture.pytorch.lenet import LeNet as LeNetTorch
from fastestimator.architecture.tensorflow.lenet import LeNet as LeNetTf
from fastestimator.dataset.data import mnist
from fastestimator.network import TFNetwork
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule.schedule import EpochScheduler, get_current_items
from fastestimator.trace import Trace


class TorchCustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}


def get_sample_tf_dataset(expand_axis=-1, batch_size=10):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_train = np.expand_dims(x_train, axis=expand_axis) / 255.0
    x_train = x_train.astype(np.float32)
    dataset_train = tf.data.Dataset.from_tensor_slices({"x": x_train, "y": y_train}).batch(batch_size)
    return dataset_train


def get_sample_torch_dataloader(expand_axis=1, batch_size=10):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_train = np.expand_dims(x_train, axis=expand_axis) / 255.0
    x_train = x_train.astype(np.float32)
    dataset_train = DataLoader(TorchCustomDataset({"x": x_train, "y": y_train}), batch_size=batch_size)

    return dataset_train


class TestEstimatorInit(unittest.TestCase):
    """ This test has dependency on:
    * fe.summary.system.System
    """
    @classmethod
    def setUpClass(cls):
        train_data, eval_data = mnist.load_data()
        cls.pipeline = fe.Pipeline(train_data=train_data)

        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        cls.network = fe.Network(ops=[
            ModelOp(model=model, inputs="x_out", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

    def test_estimator_init_check_log_step(self):
        with self.subTest("default log_steps"):
            est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)

        with self.subTest("positive log_steps"):
            est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1, log_steps=1)

        with self.subTest("negative log_steps"):
            with self.assertRaises(AssertionError):
                est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1, log_steps=-1)

    def test_estimator_init_check_monitor_name(self):
        with self.subTest("default monitor_names"):
            est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)
            self.assertEqual(est.monitor_names, {"ce"})

        with self.subTest("given monitor_names"):
            est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1, monitor_names={"a", "b"})
            self.assertEqual(est.monitor_names, {"a", "b", "ce"})

    def test_estimator_init_check_system(self):
        est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)
        self.assertIsInstance(est.system, fe.summary.System)


class TestEstimatorPrepareTraces(unittest.TestCase):
    """This test has dependency on:
    * fe.schedule.schedule.get_current_item
    """
    @classmethod
    def setUpClass(cls):
        train_data, eval_data = mnist.load_data()
        cls.pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=eval_data)

        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        cls.network = fe.Network(ops=[
            ModelOp(model=model, inputs="x_out", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

    def test_estimator_prepare_traces_check_add_trace_fit_mode(self):
        est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)
        est._prepare_traces({"train", "eval"})

        with self.subTest("check Logger"):
            self.assertIsInstance(est.traces_in_use[-1], fe.trace.Logger)

        with self.subTest("check TrainEssential"):
            self.assertIsInstance(est.traces_in_use[0], fe.trace.TrainEssential)

        with self.subTest("check EvalEssential"):
            self.assertIsInstance(est.traces_in_use[1], fe.trace.EvalEssential)

    def test_estimator_prepare_traces_check_add_trace_test_mode(self):
        est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)
        est._prepare_traces({"test"})

        with self.subTest("check Logger"):
            self.assertIsInstance(est.traces_in_use[-1], fe.trace.Logger)

        with self.subTest("check TestEssential"):
            self.assertIsInstance(est.traces_in_use[0], fe.trace.TestEssential)

    def test_estimator_prepare_traces_check_all_trace_have_system(self):
        est = fe.Estimator(pipeline=self.pipeline, network=self.network, epochs=1)
        est._prepare_traces({"train", "eval", "test"})

        for trace in get_current_items(est.traces_in_use, run_modes={"train", "eval", "test"}):
            self.assertEqual(trace.system, est.system)


class TestEstimatorConfigureLoader(unittest.TestCase):
    """This test has dependency on:
    * fe.util.util.to_tensor
    * fe.util.util.to_type
    * fe.util.util.to_shape
    """
    def test_estimator_configure_loader_torch_data_loader_tf_model(self):
        loader = get_sample_torch_dataloader()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x_out", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, max_train_steps_per_epoch=3, epochs=1)

        est.system.mode = "train"
        new_loader = est._configure_loader(loader)

        with self.subTest("check loader type"):
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.MirroredStrategy):
                self.assertIsInstance(new_loader, tf.distribute.DistributedDataset)
            else:
                self.assertIsInstance(new_loader, tf.data.Dataset)

        with self.subTest("max_train_steps_per_epoch=3"):
            iterator = iter(new_loader)
            for i in range(3):
                batch = next(iterator)

            with self.assertRaises(StopIteration):
                batch = next(iterator)

    def test_estimator_configure_loader_tf_data_loader_torch_model(self):
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTorch, optimizer_fn="adam")

        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x_out", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, max_train_steps_per_epoch=3, epochs=1)

        est.system.mode = "train"
        new_loader = est._configure_loader(loader)

        with self.subTest("check loader type"):  # it didn't change the data type
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.MirroredStrategy) and isinstance(network, TFNetwork):
                self.assertIsInstance(new_loader, tf.distribute.DistributedDataset)
            else:
                self.assertIsInstance(new_loader, tf.data.Dataset)

        with self.subTest("max_train_steps_per_epoch=3"):
            iterator = iter(new_loader)
            for i in range(3):
                batch = next(iterator)

            with self.assertRaises(StopIteration):
                batch = next(iterator)


class TestEstimatorConfigureTensor(unittest.TestCase):
    """This test has dependency on:
    * fe.util.util.to_tensor
    """
    def test_estimator_configure_tensor_tf_dataset_torch_model(self):
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTorch, optimizer_fn="adam")

        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x_out", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, max_train_steps_per_epoch=3, epochs=1)

        iterator = iter(loader)
        batch = next(iterator)
        batch = est._configure_tensor(loader, batch)
        self.assertIsInstance(batch["x"], torch.Tensor)

    def test_estimator_configure_tensor_tf_dataset_tf_model(self):
        loader = get_sample_torch_dataloader()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, max_train_steps_per_epoch=3, epochs=1)

        iterator = iter(loader)
        batch = next(iterator)
        batch = est._configure_tensor(loader, batch)
        self.assertIsInstance(batch["x"], torch.Tensor)  # _configure_tensor won't change batch from torch to tf


class TestEstimatorWarmup(unittest.TestCase):
    """This test has too many dependency to list down
    """
    def test_estimator_warmup_network_missing_key(self):
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(train_data=loader)  # "x", "y"
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        network = fe.Network(ops=[
            EpochScheduler({
                1: ModelOp(model=model, inputs="x_out", outputs="y_pred"),
                2: ModelOp(model=model, inputs="x_out", outputs="y_pred")
            })  # miss key x_out
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, epochs=2)
        est._prepare_traces(run_modes={"train", "eval"})

        # in multi-gpu environment it may raise Staging Error instead of KeyError
        with self.assertRaises((StagingError, KeyError)):
            est._warmup()

    def test_estimator_warmup_trace_missing_key(self):
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(train_data=loader)  # "x", "y"
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        network = fe.Network(ops=[
            EpochScheduler({
                1: ModelOp(model=model, inputs="x", outputs="y_pred"),
                2: ModelOp(model=model, inputs="x", outputs="y_pred")
            })  # miss key x_out
        ])

        est = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=[Trace(inputs="z")])  # miss key "z"
        est._prepare_traces(run_modes={"train", "eval"})
        with self.assertRaises(AssertionError):
            est._warmup()

    def test_estimator_warmup_tf_dataset_torch_model_smoke(self):
        loader = get_sample_tf_dataset(expand_axis=1)
        pipeline = fe.Pipeline(train_data=loader)  # "x", "y"
        model = fe.build(model_fn=LeNetTorch, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y_pred")])
        est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=[Trace(inputs="y_pred")])
        est._prepare_traces(run_modes={"train", "eval"})
        est._warmup()
        self.assertTrue(True)

    def test_estimator_warmup_torch_dataset_tf_model_smoke(self):
        loader = get_sample_torch_dataloader(expand_axis=-1)
        pipeline = fe.Pipeline(train_data=loader)  # "x", "y"
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")

        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y_pred")])

        est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=[Trace(inputs="y_pred")])
        est._prepare_traces(run_modes={"train", "eval"})
        est._warmup()
        self.assertTrue(True)


class ShoutNameOp(TensorOp):
    def __init__(self, name, iostream, inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.name = name
        self.iostream = iostream

    def forward(self, data, state):
        print("op forard: {}".format(self.name), file=self.iostream)


class ShoutNameTrace(Trace):
    def __init__(self, name, iostream, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.name = name
        self.iostream = iostream

    def on_begin(self, data) -> None:
        print("on_begin:{}".format(self.name), file=self.iostream)

    def on_epoch_begin(self, data) -> None:
        print("on_epoch_begin:{}".format(self.name), file=self.iostream)

    def on_batch_begin(self, data) -> None:
        print("on_batch_begin:{}".format(self.name), file=self.iostream)

    def on_batch_end(self, data) -> None:
        print("on_batch_end:{}".format(self.name), file=self.iostream)

    def on_epoch_end(self, data) -> None:
        print("on_epoch_end:{}".format(self.name), file=self.iostream)

    def on_end(self, data) -> None:
        print("on_end:{}".format(self.name), file=self.iostream)


class TestEstimatorFit(unittest.TestCase):
    """This test includes:
    * fe.estimator.Estimator.fit
    * fe.estimator.Estimator._start

    This test has dependency:
    * fe.estimator.Estimator._prepare_trace
    * fe.summary.system.System.reset
    """
    def test_estimator_check_network_op_trace_invoke_sequence_tf_backend(self):
        epochs = 1
        batches = 10  # dataset has 100 sample, and batch_size is 10
        iostream = StringIO()
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")
        ops = [
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            ShoutNameOp(name="A", iostream=iostream),
            ShoutNameOp(name="B", iostream=iostream)
        ]

        network = fe.Network(ops=ops)

        traces = [ShoutNameTrace(name="a", iostream=iostream), ShoutNameTrace(name="b", iostream=iostream)]
        est = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
        est.fit(warmup=False)

        # create the expected calling sequence in another iostream
        iostream2 = StringIO()
        ops2 = [ShoutNameOp(name="A", iostream=iostream2), ShoutNameOp(name="B", iostream=iostream2)]
        traces2 = [ShoutNameTrace(name="a", iostream=iostream2), ShoutNameTrace(name="b", iostream=iostream2)]

        # determine if running environment is multi-gpu (only needed for tf backend)
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            device_count = len(tf.config.list_physical_devices(device_type="GPU"))
        else:
            device_count = 1

        for trace in traces2:
            trace.on_begin(None)
        for epoch in range(epochs):
            for trace in traces2:
                trace.on_epoch_begin(None)
            for batch in range(batches):
                for trace in traces2:
                    trace.on_batch_begin(None)
                if batch == 0:
                    # ShoutOutTrace will only be invoked the number of times equal to device count while building static
                    # graph (for tf backend). ex: 4 GPU -> 4 times, CPU -> 1 time
                    for _ in range(device_count):
                        for op in ops2:
                            op.forward(None, None)
                for trace in traces2:
                    trace.on_batch_end(None)
            for trace in traces2:
                trace.on_epoch_end(None)
        for trace in traces2:
            trace.on_end(None)

        self.assertEqual(iostream.getvalue(), iostream2.getvalue())

    def test_estimator_check_network_op_trace_invoke_sequence_torch_backend(self):
        epochs = 1
        batches = 10  # dataset has 100 sample, and batch_size is 10
        iostream = StringIO()
        loader = get_sample_torch_dataloader()
        pipeline = fe.Pipeline(train_data=loader)
        model = fe.build(model_fn=LeNetTorch, optimizer_fn="adam")
        ops = [
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            ShoutNameOp(name="A", iostream=iostream),
            ShoutNameOp(name="B", iostream=iostream)
        ]

        network = fe.Network(ops=ops)

        traces = [ShoutNameTrace(name="a", iostream=iostream), ShoutNameTrace(name="b", iostream=iostream)]
        est = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
        est.fit(warmup=False)

        # create the expected calling sequence in another iostream
        iostream2 = StringIO()
        ops2 = [ShoutNameOp(name="A", iostream=iostream2), ShoutNameOp(name="B", iostream=iostream2)]
        traces2 = [ShoutNameTrace(name="a", iostream=iostream2), ShoutNameTrace(name="b", iostream=iostream2)]

        for trace in traces2:
            trace.on_begin(None)
        for epoch in range(epochs):
            for trace in traces2:
                trace.on_epoch_begin(None)
            for batch in range(batches):
                for trace in traces2:
                    trace.on_batch_begin(None)
                for op in ops2:  # ShoutOutTrace will be invoked every time (for torch backend)
                    op.forward(None, None)
                for trace in traces2:
                    trace.on_batch_end(None)
            for trace in traces2:
                trace.on_epoch_end(None)
        for trace in traces2:
            trace.on_end(None)

        self.assertEqual(iostream.getvalue(), iostream2.getvalue())


class TestEstimatorTest(unittest.TestCase):
    """This test includes:
    * fe.estimator.Estimator.test
    * fe.estimator.Estimator._start

    This test has dependency:
    * fe.estimator.Estimator._prepare_trace
    * fe.summary.system.System.reset_for_test
    """
    def test_estimator_check_network_op_trace_invoke_sequence_tf_backend(self):
        epochs = 1
        batches = 10  # dataset has 100 sample, and batch_size is 10
        iostream = StringIO()
        loader = get_sample_tf_dataset()
        pipeline = fe.Pipeline(test_data=loader)
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")
        ops = [
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            ShoutNameOp(name="A", iostream=iostream),
            ShoutNameOp(name="B", iostream=iostream)
        ]

        network = fe.Network(ops=ops)

        traces = [ShoutNameTrace(name="a", iostream=iostream), ShoutNameTrace(name="b", iostream=iostream)]
        est = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
        est.test()

        # create the expected calling sequence in another iostream
        iostream2 = StringIO()
        ops2 = [ShoutNameOp(name="A", iostream=iostream2), ShoutNameOp(name="B", iostream=iostream2)]
        traces2 = [ShoutNameTrace(name="a", iostream=iostream2), ShoutNameTrace(name="b", iostream=iostream2)]

        # determine if running environment is multi-gpu (only needed in tf backend)
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            device_count = len(tf.config.list_physical_devices(device_type="GPU"))
        else:
            device_count = 1

        for trace in traces2:
            trace.on_begin(None)
        for epoch in range(epochs):
            for trace in traces2:
                trace.on_epoch_begin(None)
            for batch in range(batches):
                for trace in traces2:
                    trace.on_batch_begin(None)
                if batch == 0:
                    # ShoutOutTrace will only be invoked the number of times equal to device count while building static
                    # graph (for tf backend). ex: 4 GPU -> 4 times, CPU -> 1 time
                    for _ in range(device_count):
                        for op in ops2:
                            op.forward(None, None)
                for trace in traces2:
                    trace.on_batch_end(None)
            for trace in traces2:
                trace.on_epoch_end(None)
        for trace in traces2:
            trace.on_end(None)

        self.assertEqual(iostream.getvalue(), iostream2.getvalue())

    def test_estimator_check_network_op_trace_invoke_sequence_torch_backend(self):
        epochs = 1
        batches = 10  # dataset has 100 sample, and batch_size is 10
        iostream = StringIO()
        loader = get_sample_torch_dataloader()
        pipeline = fe.Pipeline(test_data=loader)
        model = fe.build(model_fn=LeNetTorch, optimizer_fn="adam")
        ops = [
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            ShoutNameOp(name="A", iostream=iostream),
            ShoutNameOp(name="B", iostream=iostream)
        ]

        network = fe.Network(ops=ops)

        traces = [ShoutNameTrace(name="a", iostream=iostream), ShoutNameTrace(name="b", iostream=iostream)]
        est = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
        est.test()

        # create the expected calling sequence in another iostream
        iostream2 = StringIO()
        ops2 = [ShoutNameOp(name="A", iostream=iostream2), ShoutNameOp(name="B", iostream=iostream2)]
        traces2 = [ShoutNameTrace(name="a", iostream=iostream2), ShoutNameTrace(name="b", iostream=iostream2)]

        for trace in traces2:
            trace.on_begin(None)
        for epoch in range(epochs):
            for trace in traces2:
                trace.on_epoch_begin(None)
            for batch in range(batches):
                for trace in traces2:
                    trace.on_batch_begin(None)
                for op in ops2:  # ShoutOutTrace will be invoked every time (for torch backend)
                    op.forward(None, None)
                for trace in traces2:
                    trace.on_batch_end(None)
            for trace in traces2:
                trace.on_epoch_end(None)
        for trace in traces2:
            trace.on_end(None)

        self.assertEqual(iostream.getvalue(), iostream2.getvalue())
