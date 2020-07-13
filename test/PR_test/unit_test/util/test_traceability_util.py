import unittest

import torch

from fastestimator.util.traceability_util import FeInputSpec, _extract_args


class TestFeInputSpec(unittest.TestCase):
    def test_simple_input(self):
        inp = torch.ones(size=(32, 1, 28, 28), dtype=torch.float16)
        model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 100))
        spec = FeInputSpec(inp, model)
        result = spec.get_dummy_input()
        self.assertIsInstance(result, torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 1, 28, 28], list(result.shape), "Result shape is incorrect")
        self.assertEqual(torch.float16, result.dtype, "Result should be float16 dtype")

    def test_list_input(self):
        inp = [torch.ones(size=(32, 3, 32, 32), dtype=torch.float64), torch.ones(size=(32, 10), dtype=torch.int8)]
        model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 100))
        spec = FeInputSpec(inp, model)
        result = spec.get_dummy_input()
        self.assertIsInstance(result, list, "Spec should return a list of tensors")
        self.assertEqual(2, len(result), "Spec should return two tensors")
        self.assertIsInstance(result[0], torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 3, 32, 32], list(result[0].shape), "Result shape is incorrect")
        self.assertEqual(torch.float64, result[0].dtype, "Result dtype is incorrect")
        self.assertIsInstance(result[1], torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 10], list(result[1].shape), "Result shape is incorrect")
        self.assertEqual(torch.int8, result[1].dtype, "Result dtype is incorrect")

    def test_tuple_input(self):
        inp = (torch.ones(size=(32, 3, 32, 32), dtype=torch.float64), torch.ones(size=(32, 10), dtype=torch.int8))
        model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 100))
        spec = FeInputSpec(inp, model)
        result = spec.get_dummy_input()
        self.assertIsInstance(result, tuple, "Spec should return a tuple of tensors")
        self.assertEqual(2, len(result), "Spec should return two tensors")
        self.assertIsInstance(result[0], torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 3, 32, 32], list(result[0].shape), "Result shape is incorrect")
        self.assertEqual(torch.float64, result[0].dtype, "Result dtype is incorrect")
        self.assertIsInstance(result[1], torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 10], list(result[1].shape), "Result shape is incorrect")
        self.assertEqual(torch.int8, result[1].dtype, "Result dtype is incorrect")

    def test_set_input(self):
        inp = {torch.ones(size=(32, 3, 32, 32), dtype=torch.float64), torch.ones(size=(32, 10), dtype=torch.int8)}
        model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 100))
        spec = FeInputSpec(inp, model)
        result = spec.get_dummy_input()
        self.assertIsInstance(result, set, "Spec should return a set of tensors")
        self.assertEqual(2, len(result), "Spec should return two tensors")

    def test_map_input(self):
        inp = {'inp': torch.ones(size=(32, 3, 32, 32), dtype=torch.float64)}
        model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 100))
        spec = FeInputSpec(inp, model)
        result = spec.get_dummy_input()
        self.assertIsInstance(result, dict, "Spec should return a tuple of tensors")
        self.assertEqual(1, len(result), "Spec should return two tensors")
        self.assertIsInstance(result['inp'], torch.Tensor, "Spec should return a torch tensor")
        self.assertListEqual([32, 3, 32, 32], list(result['inp'].shape), "Result shape is incorrect")
        self.assertEqual(torch.float64, result['inp'].dtype, "Result dtype is incorrect")


class TestExtractArgs(unittest.TestCase):
    def test_single_arg(self):
        resp = _extract_args("x:")
        self.assertSetEqual({"x"}, resp)

    def test_two_args(self):
        resp = _extract_args("x, y:")
        self.assertSetEqual({"x", "y"}, resp)

    def test_default_arg(self):
        resp = _extract_args("x=5")
        self.assertSetEqual({"x"}, resp)

    def test_two_default_arg(self):
        resp = _extract_args("x,opt=5")
        self.assertSetEqual({"x", "opt"}, resp)

    def test_two_long_args(self):
        resp = _extract_args(" long_var546 ,  longer_var5932jke  : ")
        self.assertSetEqual({"long_var546", "longer_var5932jke"}, resp)

    def test_five_args(self):
        resp = _extract_args("a1, a2=3, a3, a4='watermellon', a5=b37:")
        self.assertSetEqual({'a1', 'a2', 'a3', 'a4', 'a5'}, resp)

    def test_collection_args(self):
        resp = _extract_args('x1=[a, b, c], x2 = {5: "32", 4:[21,22,23]}, x3 = (x+22):')
        self.assertSetEqual({'x1', 'x2', 'x3'}, resp)
