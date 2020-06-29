import unittest
from io import StringIO
from unittest.mock import patch

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace import Logger
from fastestimator.util.data import Data


class TestLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({})
        cls.on_begin_global_step_msg = "FastEstimator-Start: step: 2;"
        cls.on_begin_msg = "FastEstimator-Start: step: 1;"
        cls.on_batch_end_msg = "FastEstimator-Train: step: 1;"
        cls.on_epoch_end_train_msg = "FastEstimator-Train: step: 2; epoch: 0;"
        cls.on_epoch_end_eval_msg = "FastEstimator-Eval: step: 2; epoch: 0;"
        cls.on_epoch_end_test_msg = "FastEstimator-Test: step: 2; epoch: 0;"
        cls.on_end_msg = "FastEstimator-Finish: step: 2;"

    def _test_print_msg(self, func, data, msg):
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            func(data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, msg)

    def test_on_begin_global_step(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.global_step = 2
        self._test_print_msg(func=logger.on_begin, data=self.data, msg=self.on_begin_global_step_msg)

    def test_on_begin_global_step_not_none(self):
        logger = Logger()
        logger.system = sample_system_object()
        self._test_print_msg(func=logger.on_begin, data=self.data, msg=self.on_begin_msg)

    def test_on_batch_end(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.global_step = 1
        logger.system.log_steps = 3
        self._test_print_msg(func=logger.on_batch_end, data=self.data, msg=self.on_batch_end_msg)

    def test_on_epoch_end_mode_train(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.global_step = 2
        logger.system.log_steps = 3
        self._test_print_msg(func=logger.on_epoch_end, data=self.data, msg=self.on_epoch_end_train_msg)

    def test_on_epoch_end_mode_eval(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.mode = 'eval'
        logger.system.global_step = 2
        logger.system.log_steps = 3
        self._test_print_msg(func=logger.on_epoch_end, data=self.data, msg=self.on_epoch_end_eval_msg)

    def test_on_epoch_end_mode_test(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.mode = 'test'
        logger.system.global_step = 2
        logger.system.log_steps = 3
        self._test_print_msg(func=logger.on_epoch_end, data=self.data, msg=self.on_epoch_end_test_msg)

    def test_on_end(self):
        logger = Logger()
        logger.system = sample_system_object()
        logger.system.global_step = 2
        logger.system.log_steps = 3
        self._test_print_msg(func=logger.on_end, data=self.data, msg=self.on_end_msg)
