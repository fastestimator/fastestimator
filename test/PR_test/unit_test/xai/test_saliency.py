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

import numpy as np

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet


class TestSaliencyNetGetMask(unittest.TestCase):
    def test_salency_net_get_masks(self):
        outputs = "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[4, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (4, 28, 28, 1))


class TestSaliencyGetSmoothedMasks(unittest.TestCase):
    def test_salency_net_get_smoothed_masks(self):
        outputs = "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[4, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_smoothed_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (4, 28, 28, 1))


class TestSaliencyGetIntegratedMasks(unittest.TestCase):
    def test_salency_net_get_integrated_masks(self):
        outputs = "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[4, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_integrated_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (4, 28, 28, 1))
