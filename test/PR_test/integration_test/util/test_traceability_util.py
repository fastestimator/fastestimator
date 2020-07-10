import unittest

from pylatex.utils import NoEscape

from fastestimator.schedule.lr_shedule import cosine_decay
from fastestimator.util.latex_util import ContainerList
from fastestimator.util.traceability_util import _parse_lambda_fallback, _trace_value
from fastestimator.util.util import Flag


class TestTraceValue(unittest.TestCase):
    def test_simple_lambda_inlining(self):
        tables = {}
        ret_ref = Flag()
        resp = _trace_value(lambda x: x + 5, tables, ret_ref)
        self.assertEqual({}, tables, "trace_value should not have generated any tables for this lambda")
        self.assertIsInstance(resp, ContainerList, "trace_value should return a ContainerList describing the function")


class TestParseLambdaFallback(unittest.TestCase):
    def test_lambda_simple(self):
        tables = {}
        ret_ref = Flag()
        epochs = 8
        resp = _parse_lambda_fallback(
            lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3 + 1 if epochs > 2 else 1e-4),
            tables,
            ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"cosine\_decay(step, cycle\_length=3750, init\_lr=1e{-}3 + 1 if epochs > 2 else 1e{-}4)",
                         resp['function'])
        self.assertIn('kwargs', resp, "response should contain kwargs")
        self.assertIsInstance(resp['kwargs'], dict, "kwargs should be a dictionary")
        self.assertDictEqual({NoEscape('epochs'): NoEscape(r'\seqsplit{8}')}, resp['kwargs'])

    def test_lambda_inlining(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(
            lambda a, b=[0, 1, 2, 3], c={'x': "it's"}: b[0] + a * c['x'] - {0
                                                                            for j in range(5)}, tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"b{[}0{]} + a * c{[}'x'{]} {-} \{0 for j in range(5)\}", resp['function'])

    def test_nested_lambda_different_args(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda z: z / y * 5), tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda z: z / y * 5)", resp['function'])

    def test_nested_lambda_different_strings_inner(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda x, y: x(y) + 'x'), tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda x, y: x(y) + 'x')", resp['function'])

    def test_nested_lambda_fallback_strings_outer(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda x, y: x(y)) + 'x', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda x, y: x(y)) + 'x'", resp['function'])

    def test_nested_lambda_fallback_strings(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda x, y: x(y) + 'x' + 'y') + 'x', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda x, y: x(y) + 'x' + 'y') + 'x'", resp['function'])
