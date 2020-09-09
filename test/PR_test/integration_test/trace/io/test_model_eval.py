import unittest
from unittest.mock import patch

import numpy as np

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import OneLayerTorchModel, one_layer_tf_model
from fastestimator.trace import Trace
from fastestimator.trace.io.model_eval import TestCase, ModelEval
from fastestimator.util import to_number


class SampleTrace(Trace):
    """ custom trace that gets average of all samples
    """
    def on_begin(self, data):
        self.buffer = []

    def on_batch_end(self, data):
        self.buffer.append(to_number(data[self.inputs[0]]))

    def on_epoch_end(self, data):
        data.write_without_log(self.outputs[0], np.mean(np.concatenate(self.buffer)))


class TestTestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = NumpyDataset({
            "x": np.array([[1, 1, 1], [1, -1, -0.5]], dtype=np.float32), "id": np.array([0, 1], dtype=np.int32)
        })
        cls.pipeline = fe.Pipeline(test_data=dataset, batch_size=1)

    def test_instance_case_tf(self):
        test_title = "test"
        test_description = "each return needs to above 0"

        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = TestCase(description=test_description, criteria=lambda y: to_number(y) > 0, aggregate=False)
        traces = ModelEval(test_cases=test_cases, test_title=test_title, save_path="report", data_id="id")
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.model_eval.json.dump') as fake:
            estimator.test("test")
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "per-instance")

        with self.subTest("description"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("passed"):
            self.assertEqual(json_summary["tests"][0]["passed"], False)

        with self.subTest("fail_threshold"):
            self.assertEqual(json_summary["tests"][0]["fail_threshold"], 0)  # its default value should be zero

        with self.subTest("fail_number"):
            self.assertEqual(json_summary["tests"][0]["fail_number"], 1)

        with self.subTest("fail_id"):
            self.assertEqual(json_summary["tests"][0]["fail_id"], [1])

    def test_instance_case_torch(self):
        test_title = "test"
        test_description = "each return needs to above 0"

        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = TestCase(description=test_description, criteria=lambda y: to_number(y) > 0, aggregate=False)
        traces = ModelEval(test_cases=test_cases, test_title=test_title, save_path="report", data_id="id")
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.model_eval.json.dump') as fake:
            estimator.test("test")
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "per-instance")

        with self.subTest("description"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("passed"):
            self.assertEqual(json_summary["tests"][0]["passed"], False)

        with self.subTest("fail_threshold"):
            self.assertEqual(json_summary["tests"][0]["fail_threshold"], 0)  # its default value should be zero

        with self.subTest("fail_number"):
            self.assertEqual(json_summary["tests"][0]["fail_number"], 1)

        with self.subTest("fail_id"):
            self.assertEqual(json_summary["tests"][0]["fail_id"], [1])

    def test_aggregate_case_tf(self):
        test_title = "test"
        test_description = "average value of y need to be above 0"

        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = TestCase(description=test_description, criteria=lambda avg: avg > 0)
        traces = [
            SampleTrace(inputs="y", outputs="avg", mode="test"),
            ModelEval(test_cases=test_cases, test_title=test_title, save_path="report", data_id="id")
        ]
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.model_eval.json.dump') as fake:
            estimator.test("test")
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "aggregate")

        with self.subTest("description"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("passed"):
            self.assertEqual(json_summary["tests"][0]["passed"], True)

        with self.subTest("inputs"):
            self.assertEqual(json_summary["tests"][0]["inputs"], {"avg": 1.75})

    def test_aggregate_case_torch(self):
        test_title = "test"
        test_description = "average value of y need to be above 0"

        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = TestCase(description=test_description, criteria=lambda avg: avg > 0)
        traces = [
            SampleTrace(inputs="y", outputs="avg", mode="test"),
            ModelEval(test_cases=test_cases, test_title=test_title, save_path="report", data_id="id")
        ]
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.model_eval.json.dump') as fake:
            estimator.test("test")
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "aggregate")

        with self.subTest("description"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("passed"):
            self.assertEqual(json_summary["tests"][0]["passed"], True)

        with self.subTest("inputs"):
            self.assertEqual(json_summary["tests"][0]["inputs"], {"avg": 1.75})
