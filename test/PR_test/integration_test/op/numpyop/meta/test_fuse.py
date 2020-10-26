import tempfile
import unittest

import fastestimator as fe
from fastestimator.op.numpyop import NumpyOp
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch


class TestNumpyOp(NumpyOp):
    def __init__(self, inputs, outputs, mode, var):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.var = var


class TestFuse(unittest.TestCase):
    def test_save_and_load_state_tf(self):
        def instantiate_system():
            system = sample_system_object()
            system.pipeline.ops = [
                fe.op.numpyop.meta.Fuse(ops=[
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                ])
            ]
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.ops[0].ops[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.ops[0].ops[0].var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            system.pipeline.ops = [
                fe.op.numpyop.meta.Fuse(ops=[
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                    TestNumpyOp(inputs="x", outputs="x", mode="train", var=1),
                ])
            ]
            return system

        system = instantiate_system()

        # make some changes
        new_var = 2
        system.pipeline.ops[0].ops[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.ops[0].ops[0].var

        self.assertEqual(loaded_var, new_var)
