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
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydot

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import OneLayerTorchModel, one_layer_tf_model
from fastestimator.trace import Trace
from fastestimator.trace.io.test_report import TestCase, TestReport
from fastestimator.util import to_number


def _lacks_graphviz():
    try:
        pydot.Dot.create(pydot.Dot())
    except OSError:
        return True
    return False


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
        cls.pipeline = fe.Pipeline(test_data=dataset, batch_size=1, num_process=0)

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_instance_case_tf(self):
        test_title = "test"
        test_description = "each return needs to above 0"
        test_description2 = "each return needs to above -10"
        save_path = tempfile.mkdtemp()
        exp_name = "exp"

        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = [
            TestCase(description=test_description,
                     criteria=lambda y: to_number(y) > 10,
                     aggregate=False,
                     fail_threshold=1),
            TestCase(description=test_description2, criteria=lambda y: to_number(y) > -10, aggregate=False)
        ]
        traces = TestReport(test_cases=test_cases, test_title=test_title, save_path=save_path, data_id="id")
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.test_report.json.dump') as fake:
            estimator.test(exp_name)
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type 1"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "per-instance")

        with self.subTest("test_type 2"):
            self.assertEqual(json_summary["tests"][1]["test_type"], "per-instance")

        with self.subTest("description 1"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("description 2"):
            self.assertEqual(json_summary["tests"][1]["description"], test_description2)

        with self.subTest("passed 1"):
            self.assertEqual(json_summary["tests"][0]["passed"], False)

        with self.subTest("passed 2"):
            self.assertEqual(json_summary["tests"][1]["passed"], True)

        with self.subTest("fail_threshold 1"):
            self.assertEqual(json_summary["tests"][0]["fail_threshold"], 1)

        with self.subTest("fail_threshold 2"):
            self.assertEqual(json_summary["tests"][1]["fail_threshold"], 0)  # its default value should be zero

        with self.subTest("fail_number 1"):
            self.assertEqual(json_summary["tests"][0]["fail_number"], 2)

        with self.subTest("fail_number 2"):
            self.assertEqual(json_summary["tests"][1]["fail_number"], 0)

        with self.subTest("fail_id 1"):
            self.assertEqual(json_summary["tests"][0]["fail_id"], [0, 1])

        with self.subTest("fail_id 2"):
            self.assertEqual(json_summary["tests"][1]["fail_id"], [])

        with self.subTest("check pdf report"):
            report_path = os.path.join(save_path, exp_name + "_TestReport.pdf")
            self.assertTrue(os.path.exists(report_path))

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_instance_case_torch(self):
        test_title = "test"
        test_description = "each return needs to above 0"
        test_description2 = "each return needs to above -10"
        save_path = tempfile.mkdtemp()
        exp_name = "exp"

        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = [
            TestCase(description=test_description,
                     criteria=lambda y: to_number(y) > 10,
                     aggregate=False,
                     fail_threshold=1),
            TestCase(description=test_description2, criteria=lambda y: to_number(y) > -10, aggregate=False)
        ]
        traces = TestReport(test_cases=test_cases, test_title=test_title, save_path=save_path, data_id="id")
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.test_report.json.dump') as fake:
            estimator.test(exp_name)
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type 1"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "per-instance")

        with self.subTest("test_type 2"):
            self.assertEqual(json_summary["tests"][1]["test_type"], "per-instance")

        with self.subTest("description 1"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("description 2"):
            self.assertEqual(json_summary["tests"][1]["description"], test_description2)

        with self.subTest("passed 1"):
            self.assertEqual(json_summary["tests"][0]["passed"], False)

        with self.subTest("passed 2"):
            self.assertEqual(json_summary["tests"][1]["passed"], True)

        with self.subTest("fail_threshold 1"):
            self.assertEqual(json_summary["tests"][0]["fail_threshold"], 1)

        with self.subTest("fail_threshold 2"):
            self.assertEqual(json_summary["tests"][1]["fail_threshold"], 0)  # its default value should be zero

        with self.subTest("fail_number 1"):
            self.assertEqual(json_summary["tests"][0]["fail_number"], 2)

        with self.subTest("fail_number 2"):
            self.assertEqual(json_summary["tests"][1]["fail_number"], 0)

        with self.subTest("fail_id 1"):
            self.assertEqual(json_summary["tests"][0]["fail_id"], [0, 1])

        with self.subTest("fail_id 2"):
            self.assertEqual(json_summary["tests"][1]["fail_id"], [])

        with self.subTest("check pdf report"):
            report_path = os.path.join(save_path, exp_name + "_TestReport.pdf")
            self.assertTrue(os.path.exists(report_path))

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_aggregate_case_tf(self):
        test_title = "test"
        test_description = "average value of y need to be above 0"
        test_description2 = "average value of y need to be above 100"
        save_path = tempfile.mkdtemp()
        exp_name = "exp"

        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = [
            TestCase(description=test_description, criteria=lambda avg: avg > 0),
            TestCase(description=test_description2, criteria=lambda avg: avg > 100)
        ]
        traces = [
            SampleTrace(inputs="y", outputs="avg", mode="test"),
            TestReport(test_cases=test_cases, test_title=test_title, save_path=save_path, data_id="id")
        ]
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.test_report.json.dump') as fake:
            estimator.test(exp_name)
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type 1"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "aggregate")
        with self.subTest("test_type 2"):
            self.assertEqual(json_summary["tests"][1]["test_type"], "aggregate")

        with self.subTest("description 1"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("description 2"):
            self.assertEqual(json_summary["tests"][1]["description"], test_description2)

        with self.subTest("passed 1"):
            self.assertEqual(json_summary["tests"][0]["passed"], True)

        with self.subTest("passed 2"):
            self.assertEqual(json_summary["tests"][1]["passed"], False)

        with self.subTest("inputs 1"):
            self.assertEqual(json_summary["tests"][0]["inputs"], {"avg": 1.75})

        with self.subTest("inputs 2"):
            self.assertEqual(json_summary["tests"][1]["inputs"], {"avg": 1.75})

        with self.subTest("check pdf report"):
            report_path = os.path.join(save_path, exp_name + "_TestReport.pdf")
            self.assertTrue(os.path.exists(report_path))

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_aggregate_case_torch(self):
        test_title = "test"
        test_description = "average value of y need to be above 0"
        test_description2 = "average value of y need to be above 100"
        save_path = tempfile.mkdtemp()
        exp_name = "exp"

        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        test_cases = [
            TestCase(description=test_description, criteria=lambda avg: avg > 0),
            TestCase(description=test_description2, criteria=lambda avg: avg > 100)
        ]
        traces = [
            SampleTrace(inputs="y", outputs="avg", mode="test"),
            TestReport(test_cases=test_cases, test_title=test_title, save_path=save_path, data_id="id")
        ]
        estimator = fe.Estimator(pipeline=self.pipeline, network=network, epochs=1, traces=traces)

        with patch('fastestimator.trace.io.test_report.json.dump') as fake:
            estimator.test(exp_name)
            json_summary = fake.call_args[0][0]

        with self.subTest("title"):
            self.assertEqual(json_summary["title"], test_title)

        with self.subTest("timestamp"):
            self.assertIn("timestamp", json_summary)

        with self.subTest("execution_time(s)"):
            self.assertIn("execution_time(s)", json_summary)

        with self.subTest("test_type 1"):
            self.assertEqual(json_summary["tests"][0]["test_type"], "aggregate")
        with self.subTest("test_type 2"):
            self.assertEqual(json_summary["tests"][1]["test_type"], "aggregate")

        with self.subTest("description 1"):
            self.assertEqual(json_summary["tests"][0]["description"], test_description)

        with self.subTest("description 2"):
            self.assertEqual(json_summary["tests"][1]["description"], test_description2)

        with self.subTest("passed 1"):
            self.assertEqual(json_summary["tests"][0]["passed"], True)

        with self.subTest("passed 2"):
            self.assertEqual(json_summary["tests"][1]["passed"], False)

        with self.subTest("inputs 1"):
            self.assertEqual(json_summary["tests"][0]["inputs"], {"avg": 1.75})

        with self.subTest("inputs 2"):
            self.assertEqual(json_summary["tests"][1]["inputs"], {"avg": 1.75})

        with self.subTest("check pdf report"):
            report_path = os.path.join(save_path, exp_name + "_TestReport.pdf")
            self.assertTrue(os.path.exists(report_path))
