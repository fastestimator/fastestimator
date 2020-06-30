import unittest
from fastestimator.architecture.tensorflow import LeNet
import numpy as np
import fastestimator as fe

class TestSaliencyNetGetMask(unittest.TestCase):
    def test_salency_net_get_masks(self):
        outputs= "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[1, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (1, 28, 28, 1))


class TestSaliencyGetSmoothedMasks(unittest.TestCase):
    def test_salency_net_get_smoothed_masks(self):
        outputs = "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[1, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_smoothed_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (1, 28, 28, 1))


class TestSaliencyGetIntegratedMasks(unittest.TestCase):
    def test_salency_net_get_integrated_masks(self):
        outputs = "saliency"
        batch = {"x": np.random.uniform(0, 1, size=[1, 28, 28, 1]).astype(np.float32)}

        model = fe.build(model_fn=LeNet, optimizer_fn="adam")
        saliency = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred", outputs=outputs)
        new_batch = saliency.get_integrated_masks(batch)

        with self.subTest("check outputs exist"):
            self.assertIn(outputs, new_batch)

        with self.subTest("check output size"):
            self.assertEqual(new_batch[outputs].numpy().shape, (1, 28, 28, 1))