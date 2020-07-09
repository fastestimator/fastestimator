import unittest
from copy import deepcopy

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet as LeNetTorch
from fastestimator.architecture.tensorflow import LeNet as LeNetTf
from fastestimator.network import TFNetwork, TorchNetwork
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy, MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, RepeatScheduler
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model


class UnknownCompiledModel:
    def __init__(self, model):
        self.model = model
        self.fe_compiled = True


class SampleNumpyOp(NumpyOp):
    def forward(self, data, state):
        return data


class SampleTensorOp(TensorOp):
    def forward(self, data, state):
        return data


def get_torch_lenet_model_weight(model):
    weight = []
    weight.append(deepcopy(model.conv1.weight.data.numpy()))
    weight.append(deepcopy(model.conv2.weight.data.numpy()))
    weight.append(deepcopy(model.conv3.weight.data.numpy()))
    weight.append(deepcopy(model.fc1.weight.data.numpy()))
    weight.append(deepcopy(model.fc1.weight.data.numpy()))

    return weight


def get_tf_model_weight(model):
    weight = []
    for layer in model.layers:
        weight.append(layer.get_weights())

    return weight


class TestNetworkCollectModel(unittest.TestCase):
    """This test has dependency on:
    * fe.schedule.schedule.get_current_items
    * fe.op.tensorop.model.model.ModelOp
    * fe.op.tensorop.model.update.UpdateOp
    * fe.network.build
    """
    def test_network_collect_model_with_model_op_and_update_op(self):
        model_fns = {"tf_model_fn": one_layer_tf_model, "torch_model_fn": OneLayerTorchModel}

        for name, model_fn in model_fns.items():
            with self.subTest(name):
                model = fe.build(model_fn=model_fn, optimizer_fn=None)
                model2 = fe.build(model_fn=model_fn, optimizer_fn=None)
                ops = [
                    SampleNumpyOp(inputs="x", outputs="x"),
                    SampleTensorOp(inputs="x", outputs="x"),
                    ModelOp(model=model, inputs="x", outputs="y2"),
                    UpdateOp(model=model2, loss_name="ce")
                ]

                models = fe.network._collect_models(ops)
                ans = {model, model2}
                self.assertEqual(models, ans)

    def test_network_collect_model_no_model_op_and_update_op(self):
        ops = [
            SampleNumpyOp(inputs="x", outputs="x"),
            SampleTensorOp(inputs="x", outputs="x"),
        ]

        models = fe.network._collect_models(ops)
        ans = set()
        self.assertEqual(models, ans)


class TestNetworkNetwork(unittest.TestCase):
    """This test has dependency:
    * fe.op.tensorop.model.model.ModelOp
    * fe.network.build
    """
    @classmethod
    def setUpClass(cls):
        cls.tf_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn=None)
        cls.torch_model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=None)
        cls.unknown_model = UnknownCompiledModel("string")

    def test_network_network_case_could_work(self):
        ops_dict = {
            "single tf model": [ModelOp(model=self.tf_model, inputs="x", outputs="y")],
            "multiple tf model": [
                ModelOp(model=self.tf_model, inputs="x", outputs="y"),
                ModelOp(model=self.tf_model, inputs="x", outputs="y")
            ]
        }

        for case, ops in ops_dict.items():
            with self.subTest(case):
                network = fe.Network(ops=ops)
                self.assertIsInstance(network, TFNetwork)

        ops_dict = {
            "single torch model": [ModelOp(model=self.torch_model, inputs="x", outputs="y")],
            "multiple torch model": [
                ModelOp(model=self.torch_model, inputs="x", outputs="y"),
                ModelOp(model=self.torch_model, inputs="x", outputs="y")
            ],
        }

        for case, ops in ops_dict.items():
            with self.subTest(case):
                network = fe.Network(ops=ops)
                self.assertIsInstance(network, TorchNetwork)

    def test_network_network_case_could_not_work(self):
        ops_dict = {
            "no model": [SampleTensorOp(inputs="x", outputs="y")],
            "mixed model type": [
                ModelOp(model=self.torch_model, inputs="x", outputs="y"),
                ModelOp(model=self.tf_model, inputs="x", outputs="y")
            ]
        }

        for case, ops in ops_dict.items():
            with self.subTest(case):
                with self.assertRaises(AssertionError):
                    network = fe.Network(ops=ops)

    def test_network_network_unkown_compiled_model(self):
        with self.assertRaises(ValueError):
            network = fe.Network(ops=[ModelOp(model=self.unknown_model, inputs="x", outputs="y")])


class TestNetworkBuildOptimizer(unittest.TestCase):
    """This test includes:
    * fe.network._build_optimizer
    * fe.network._optimizer_fn_from_string
    * fe.network._optimizer_fn_to_optimizer
    """
    @classmethod
    def setUpClass(cls):
        cls.tf_model = one_layer_tf_model()
        cls.torch_model = OneLayerTorchModel()

    def test_network_build_optimizer_tf_model_optimizer_str(self):
        str_list = ['adadelta', 'adagrad', 'adam', 'adamax', 'rmsprop', 'sgd']
        for opt_name in str_list:
            with self.subTest(optimizer_fn=opt_name):
                optimizer = fe.network._build_optimizer(optimizer_fn=opt_name, model=self.tf_model, framework="tf")
                self.assertIsInstance(optimizer, tf.optimizers.Optimizer)

    def test_network_build_optimizer_torch_model_optimizer_str(self):
        str_list = ['adadelta', 'adagrad', 'adam', 'adamax', 'rmsprop', 'sgd']
        for opt_name in str_list:
            with self.subTest(optimizer_fn=opt_name):
                optimizer = fe.network._build_optimizer(optimizer_fn=opt_name,
                                                        model=self.torch_model,
                                                        framework="torch")
                self.assertIsInstance(optimizer, torch.optim.Optimizer)

    def test_network_build_optimizer_tf_model_optimizer_fn(self):
        fn_list = [tf.optimizers.Adadelta, lambda: tf.optimizers.Adam(lr=0.001)]
        for opt_fn in fn_list:
            with self.subTest(optimizer_fn=opt_fn):
                optimizer = fe.network._build_optimizer(optimizer_fn=opt_fn, model=self.tf_model, framework="tf")
                self.assertIsInstance(optimizer, tf.optimizers.Optimizer)

    def test_network_build_optimizer_torch_model_optimizer_fn(self):
        opt_fn = lambda x: torch.optim.SGD(params=x, lr=0.01)
        optimizer = fe.network._build_optimizer(optimizer_fn=opt_fn, model=self.torch_model, framework="torch")
        self.assertIsInstance(optimizer, torch.optim.Optimizer)


class TestNetworkFeCompile(unittest.TestCase):
    """This test has dependency on:
    * fe.network._build_optimizer
    * fe.schedule.schedule.EpochScheduler
    * fe.schedule.schedule.RepeatScheduler
    """
    @classmethod
    def setUpClass(cls):
        cls.tf_model = one_layer_tf_model()
        cls.torch_model = OneLayerTorchModel()

    def test_network_fe_compile_optimizer_epochscheduler_tf_check_load_wight(self):
        with unittest.mock.patch("fastestimator.network.load_model") as fake:
            optimizer = EpochScheduler(epoch_dict={1: "adam", 10: "sgd"})
            model = fe.network._fe_compile(model=self.tf_model,
                                           optimizer_fn=optimizer,
                                           weight="example_path",
                                           name="test",
                                           framework="tf")

            _, weight = fake.call_args[0]
            self.assertEqual(weight, "example_path")

    def test_network_fe_compile_optimizer_epochscheduler_tf_check_all(self):
        optimizer = EpochScheduler(epoch_dict={1: "adam", 10: "sgd"})
        model = fe.network._fe_compile(model=self.tf_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name="test",
                                       framework="tf")

        with self.subTest("check optimizer instantiation"):
            for optimizer in model.optimizer.get_all_values():
                self.assertIsInstance(optimizer, tf.optimizers.Optimizer)

        with self.subTest("check current_optimizer"):
            self.assertIsInstance(model.current_optimizer, tf.optimizers.Adam)

        with self.subTest("check model_name"):
            self.assertEqual(model.model_name, "test")

        with self.subTest("check fe_compiled"):
            self.assertEqual(model.fe_compiled, True)

    def test_network_fe_compile_optimizer_repeatscheduler_tf_check_optimizer(self):
        optimizer = RepeatScheduler(["adam", "sgd"])
        model = fe.network._fe_compile(model=self.tf_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="tf")

        with self.subTest("check optimizer instantiation"):
            for optimizer in model.optimizer.get_all_values():
                self.assertIsInstance(optimizer, tf.optimizers.Optimizer)

        with self.subTest("check current optimizer"):
            self.assertIsInstance(model.current_optimizer, tf.optimizers.Adam)

    def test_network_fe_compile_optimizer_no_scheduler_tf_check_optimizer(self):
        optimizer = "adam"
        model = fe.network._fe_compile(model=self.tf_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="tf")
        with self.subTest("check optimizer instantiation"):
            self.assertIsInstance(model.optimizer, tf.optimizers.Optimizer)

        with self.subTest("check current optimizer"):
            self.assertEqual(model.current_optimizer, model.optimizer)

    def test_network_fe_compile_optimizer_epochscheduler_torch_check_optimizer(self):
        optimizer = EpochScheduler(epoch_dict={1: "adam", 10: "sgd"})
        model = fe.network._fe_compile(model=self.torch_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="torch")

        with self.subTest("check optimizer instantiation"):
            for optimizer in model.optimizer.get_all_values():
                self.assertIsInstance(optimizer, torch.optim.Optimizer)

        with self.subTest("check current optimizer"):
            self.assertIsInstance(model.current_optimizer, torch.optim.Adam)

    def test_network_fe_compile_optimizer_repeatscheduler_torch_check_optimizer(self):
        optimizer = RepeatScheduler(["adam", "sgd"])
        model = fe.network._fe_compile(model=self.torch_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="torch")

        with self.subTest("check optimizer instantiation"):
            for optimizer in model.optimizer.get_all_values():
                self.assertIsInstance(optimizer, torch.optim.Optimizer)

        with self.subTest("check current optimizer"):
            self.assertIsInstance(model.current_optimizer, torch.optim.Adam)

    def test_network_fe_compile_optimizer_no_scheduler_torch_check_optimizer(self):
        optimizer = "adam"
        model = fe.network._fe_compile(model=self.torch_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="torch")
        with self.subTest("check optimizer instantiation"):
            self.assertIsInstance(model.optimizer, torch.optim.Optimizer)

        with self.subTest("check current optimizer"):
            self.assertEqual(model.current_optimizer, model.optimizer)

        model = fe.network._fe_compile(model=self.torch_model,
                                       optimizer_fn=optimizer,
                                       weight=None,
                                       name=None,
                                       framework="torch")


class TestNetworkBuild(unittest.TestCase):
    """This test has dependency on:
    * fe.util.traceability_util.trace_model
    * fe.network._fe_compile
    """
    def test_network_build_check_model_name(self):
        with self.subTest("not specify model_name"):
            model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
            model2 = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
            self.assertNotEqual(model.model_name, model2.model_name)

        with self.subTest("specify model_name"):
            model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam", model_name="test")
            self.assertEqual(model.model_name, "test")

    def test_network_build_tf_model_tf_optimizer_check_model_optimizer_instance(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn=tf.optimizers.Adadelta)
        with self.subTest("check model instance"):
            self.assertIsInstance(model, tf.keras.Model)

        with self.subTest("check optimizer"):
            self.assertIsInstance(model.optimizer, tf.optimizers.Optimizer)

    def test_network_build_torch_model_torch_optimizer_check_model_optimizer_instance(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01))
        with self.subTest("check model instance"):
            self.assertIsInstance(model, torch.nn.Module)

        with self.subTest("check optimizer"):
            self.assertIsInstance(model.optimizer, torch.optim.Optimizer)

    def test_network_build_tf_model_torch_optimizer_check_assertion_error(self):
        with self.assertRaises(AssertionError):
            model = fe.build(model_fn=one_layer_tf_model, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01))

    def test_network_build_torch_model_tf_optimizer_check_assertion_error(self):
        with self.subTest("optimizer_fn directly uses tf optimizer "):
            with self.assertRaises(AssertionError):
                model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=tf.optimizers.Adadelta)

        with self.subTest("optimizer_fn use lambda function"):
            with self.assertRaises(ValueError):
                model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=lambda: tf.optimizers.Adadelta())

    def test_network_build_unknown_model_check_assertion_error(self):
        with self.assertRaises(ValueError):
            model = fe.build(model_fn=lambda: "string", optimizer_fn=tf.optimizers.Adadelta)

    def test_network_build_check_load_weight_from_path(self):
        with unittest.mock.patch("fastestimator.network.load_model") as fake:
            optimizer = EpochScheduler(epoch_dict={1: "adam", 10: "sgd"})
            model = fe.build(model_fn=one_layer_tf_model,
                             optimizer_fn=tf.optimizers.Adadelta,
                             weights_path="example_path")

            _, weight = fake.call_args[0]
            self.assertEqual(weight, "example_path")


class TestNetworkTransform(unittest.TestCase):
    """This test includes:
    * fe.network.TFNetwork.transform (and its all invoking function)
    * fe.network.TorchNetwork.transform (and its all invoking function)
    """
    def test_network_transform_one_layer_model_tf(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        weight = get_tf_model_weight(model)
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            MeanSquaredError(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])
        batch = {"x": np.array([[1, 1, 1]]), "y": np.array([[1]])}
        batch = network.transform(data=batch, mode="train")

        with self.subTest("output y_pred check"):
            ans = np.array([[6]], dtype=np.float32)  # 1*1 + 1*2 + 1*3
            self.assertTrue(np.array_equal(batch["y_pred"].numpy(), ans))

        with self.subTest("output ce check"):
            self.assertEqual(batch["ce"].numpy(), 25)  # (6-1)^2

        with self.subTest("check whether model weight changed"):
            weight2 = get_tf_model_weight(model)
            self.assertFalse(is_equal(weight, weight2))

    def test_network_transform_lenet_tf(self):
        model = fe.build(model_fn=LeNetTf, optimizer_fn="adam")
        weight = get_tf_model_weight(model)
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        batch = {"x": np.ones((1, 28, 28, 1)), "y": np.array([[1]])}
        batch = network.transform(data=batch, mode="train")
        with self.subTest("output y_pred check"):
            self.assertTrue("y_pred" in batch.keys())
            self.assertIsNotNone(batch["y_pred"])

        with self.subTest("output ce check"):
            self.assertTrue("ce" in batch.keys())
            self.assertIsNotNone(batch["ce"])

        with self.subTest("check whether model weight changed"):
            weight2 = get_tf_model_weight(model)
            self.assertFalse(is_equal(weight, weight2))

    def test_network_transform_lenet_torch(self):
        model = fe.build(model_fn=LeNetTorch, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1.0))
        weight = get_torch_lenet_model_weight(model)
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])
        batch = {"x": np.ones((1, 1, 28, 28), dtype=np.float32), "y": np.array([[1]], dtype=np.float32)}
        batch = network.transform(data=batch, mode="train")

        with self.subTest("output y_pred check"):
            self.assertTrue("y_pred" in batch.keys())
            self.assertIsNotNone(batch["y_pred"])

        with self.subTest("output ce check"):
            self.assertTrue("ce" in batch.keys())
            self.assertIsNotNone(batch["ce"])

        with self.subTest("check whether model weight changed"):
            weight2 = get_torch_lenet_model_weight(model)
            self.assertFalse(is_equal(weight, weight2))
