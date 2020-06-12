import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestData(unittest.TestCase):
    def setUp(self):
        self.d = fe.util.Data({"a": 0, "b": 1, "c": 2})

    def test_write_with_log(self):
        self.d.write_with_log("d", 3)
        self.assertEqual(self.d.read_logs(), {'d': 3})

    def test_write_without_log(self):
        self.d.write_without_log("e", 5)
        self.assertEqual(self.d.read_logs(), {})

    def test_read_logs(self):
        self.d.write_with_log("d", 3)
        self.d.write_with_log("a", 4)
        self.assertEqual(self.d.read_logs(), {"d": 3, "a": 4})
