# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

import fastestimator as fe
import tensorflow as tf
import torch
from fastestimator.test.unittest_util import is_equal


class TestGetGradient(unittest.TestCase):
    def test_get_gradient_tf_tensor_higher_order_false(self):
        def check_gradient(x):
            with tf.GradientTape(persistent=True) as tape:
                y = x * x
                with self.subTest("check gradient"):
                    obj1 = fe.backend.get_gradient(target=y, sources=x, tape=tape)  # [2.0, 4.0, 6.0]
                    obj2 = tf.constant([2.0, 4.0, 6.0])
                    self.assertTrue(is_equal(obj1, obj2))

                with self.subTest("check gradient of gradient"):
                    obj1 = fe.backend.get_gradient(target=obj1, sources=x, tape=tape)  # None
                    self.assertTrue(is_equal(obj1, None))

        x = tf.Variable([1.0, 2.0, 3.0])
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            strategy.run(check_gradient, args=(x, ))
        else:
            check_gradient(x)

    def test_get_gradient_tf_tensor_higher_order_true(self):
        def check_gradient(x):
            with tf.GradientTape(persistent=True) as tape:
                y = x * x
                with self.subTest("check gradient"):
                    obj1 = fe.backend.get_gradient(target=y, sources=x, tape=tape, higher_order=True)  # [2.0, 4.0, 6.0]
                    obj2 = tf.constant([2.0, 4.0, 6.0])
                    self.assertTrue(is_equal(obj1, obj2))

                with self.subTest("check gradient of gradient"):
                    obj1 = fe.backend.get_gradient(target=obj1, sources=x, tape=tape)  # [2.0, 2.0, 2.0]
                    obj2 = tf.constant([2.0, 2.0, 2.0])
                    self.assertTrue(is_equal(obj1, obj2))

        x = tf.Variable([1.0, 2.0, 3.0])
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            strategy.run(check_gradient, args=(x, ))
        else:
            check_gradient(x)

    def test_get_gradient_torch_tensor_higher_order_false(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        obj1 = fe.backend.get_gradient(target=y, sources=x)  # [2.0, 4.0, 6.0]
        obj2 = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(is_equal(obj1, obj2))

    def test_get_gradient_torch_tensor_higher_order_true(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        with self.subTest("check gradient"):
            obj1 = fe.backend.get_gradient(target=y, sources=x, higher_order=True)  # [2.0, 4.0, 6.0]
            obj2 = torch.tensor([2.0, 4.0, 6.0])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("check gradient of gradient"):
            obj1 = fe.backend.get_gradient(target=obj1, sources=x)  # [2.0, 2.0, 2.0]
            obj2 = torch.tensor([2.0, 2.0, 2.0])
            self.assertTrue(is_equal(obj1, obj2))
