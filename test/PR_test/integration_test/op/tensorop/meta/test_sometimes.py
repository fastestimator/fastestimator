import tempfile
import unittest

import fastestimator as fe
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch


class TestTensorOp(TensorOp):
    def __init__(self, inputs, outputs, mode, var):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.var = var


class TestSometimes(unittest.TestCase):
    def test_save_and_load_state_tf(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(ops=[
                fe.op.tensorop.meta.Sometimes(TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1)),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.network.ops[0].op.var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.ops[0].op.var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
            system.network = fe.Network(ops=[
                fe.op.tensorop.meta.Sometimes(TestTensorOp(inputs="x_out", outputs="x_out", mode="train", var=1)),
                ModelOp(model=model, inputs="x_out", outputs="y_pred")
            ])
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.network.ops[0].op.var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.ops[0].op.var

        self.assertEqual(loaded_var, new_var)
