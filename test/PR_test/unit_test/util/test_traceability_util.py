import unittest

from fastestimator.util.traceability_util import _extract_args


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
