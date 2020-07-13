import unittest

import numpy as np
from pylatex.utils import NoEscape

from fastestimator.schedule.lr_shedule import cosine_decay
from fastestimator.util.latex_util import ContainerList
from fastestimator.util.traceability_util import _parse_lambda, _parse_lambda_fallback, _trace_value
from fastestimator.util.util import Flag


class TestTraceValue(unittest.TestCase):
    def test_simple_lambda_inlining(self):
        tables = {}
        ret_ref = Flag()
        resp = _trace_value(lambda x: x + 5, tables, ret_ref)
        self.assertEqual({}, tables, "trace_value should not have generated any tables for this lambda")
        self.assertIsInstance(resp, ContainerList, "trace_value should return a ContainerList describing the function")


class TestParseLambda(unittest.TestCase):
    def test_conditional_lambda(self):
        tables = {}
        ret_ref = Flag()
        a = 5
        resp = _parse_lambda(
            lambda x: [0, 1] if x > 10 else (1, a) if x > 8 else {1, 3} if x > 6 else {1: 5} if x < 0 else {
                'key': 0, 'key2': 1
            },
            tables,
            ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertIsInstance(resp['function'], ContainerList, "_parse_lambda should return a ContainerList")


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

    def test_nested_lambda_different_strings_outer(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda x, y: x(y)) + 'x', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda x, y: x(y)) + 'x'", resp['function'])

    def test_nested_lambda_different_strings(self):
        tables = {}
        ret_ref = Flag()
        resp = _parse_lambda_fallback(lambda x, y: x(lambda x, y: x(y) + 'x' + 'y') + 'x', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x(lambda x, y: x(y) + 'x' + 'y') + 'x'", resp['function'])

    def test_multi_lambda_different_args(self):
        tables = {}
        ret_ref = Flag()
        resp, other = _parse_lambda_fallback(lambda x: x + 5, tables, ret_ref), lambda y: y + 5
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x + 5", resp['function'])

    def test_multi_lambda_different_strings(self):
        tables = {}
        ret_ref = Flag()
        other, resp = lambda x: x + 'x1', _parse_lambda_fallback(lambda x: x + 'x2', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x + 'x2'", resp['function'])

    def test_multi_lambda_same_fn(self):
        tables = {}
        ret_ref = Flag()
        other, resp = lambda x: x + 'x1', _parse_lambda_fallback(lambda x: x + 'x1', tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"x + 'x1'", resp['function'])

    def test_multi_lambda_different_refs(self):
        tables = {}
        ret_ref = Flag()
        other, resp = lambda x: np.log2(128) + x, _parse_lambda_fallback(lambda x: np.ceil(128) + x, tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"np.ceil(128) + x", resp['function'])

    def test_multi_lambda_different_vars(self):
        tables = {}
        ret_ref = Flag()
        a = 5
        b = 10
        other, resp = lambda x: a * x, _parse_lambda_fallback(lambda x: b * x, tables, ret_ref)
        self.assertIsInstance(resp, dict, "_parse_lambda_fallback should return a dictionary")
        self.assertEqual({}, tables, "_parse_lambda_fallback should not have generated any tables for this lambda")
        self.assertIn('function', resp, "response should contain a function summary")
        self.assertEqual(r"b * x", resp['function'])
