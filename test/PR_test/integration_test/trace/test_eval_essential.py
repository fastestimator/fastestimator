import unittest

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace import EvalEssential
from fastestimator.util.data import Data


class TestEvalEssential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})

    def test_on_epoch_begin(self):
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.on_epoch_begin(data=self.data)
        self.assertIsNone(eval_essential.eval_results)

    def test_on_batch_end_eval_results_not_none(self):
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.eval_results = {'loss': [95]}
        eval_essential.on_batch_end(data=self.data)
        self.assertEqual(eval_essential.eval_results['loss'], [95, 10])

    def test_on_batch_end_eval_results_none(self):
        data = Data({'loss': 5})
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.on_batch_end(data=data)
        self.assertEqual(eval_essential.eval_results['loss'], [5])

    def test_on_epoch_end(self):
        data = Data({})
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.eval_results = {'loss': [10, 20]}
        eval_essential.on_epoch_end(data=data)
        self.assertEqual(data['loss'], 15.0)
