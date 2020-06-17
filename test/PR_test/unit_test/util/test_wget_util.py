import io
import sys
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import torch
import wget

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestWgetUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = " 10% [......                                                        ] 0.00 / 0.00 MB"

    def test_bar_custom(self):
        self.assertEqual(fe.util.wget_util.bar_custom(10, 100), self.op)
