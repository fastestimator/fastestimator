import os
import tempfile
import unittest
from unittest.mock import Mock

import fastestimator as fe


class TestSystem(unittest.TestCase):
    def test_system_save_state_load_state(self):
        system = fe.summary.System(pipeline=Mock(), network=Mock(), traces=Mock())
        global_step = 100
        epoch_idx = 10
        file_path = os.path.join(tempfile.mkdtemp(), "test.json")
        system.global_step = global_step
        system.epoch_idx = epoch_idx

        with self.subTest("check save_state dumped file"):
            system.save_state(file_path)
            self.assertTrue(os.path.exists(file_path))

        system.global_step = 0
        system.epoch_idx = 0
        with self.subTest("check state after load_state"):
            system.load_state(file_path)
            self.assertEqual(system.global_step, global_step)
            self.assertEqual(system.epoch_idx, epoch_idx)
