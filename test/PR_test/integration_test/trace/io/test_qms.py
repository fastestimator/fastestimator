import os
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import is_equal, one_layer_tf_model
from fastestimator.trace import Trace
from fastestimator.trace.io import QMSTest


class PassTrace(Trace):
    def __init__(self, inputs, mode):
        super().__init__(inputs=inputs, outputs=inputs, mode=mode)

    def on_epoch_begin(self, data) -> None:
        self.record = {x: [] for x in self.inputs}

    def on_batch_end(self, data) -> None:
        for name in self.inputs:
            self.record[name].append(deepcopy(data[name].numpy()))

    def on_epoch_end(self, data) -> None:
        for name in self.inputs:
            data.write_without_log(name, self.record[name])


def get_sample_tf_dataset():
    x_train = np.array([[1, 1, 1], [1, 2, -3]], dtype=np.float32)
    dataset_train = tf.data.Dataset.from_tensor_slices({"x": x_train}).batch(2)
    return dataset_train


class TestQMSTest(unittest.TestCase):
    def test_qms(self):
        test_data = get_sample_tf_dataset()
        pipeline = fe.Pipeline(test_data=test_data)
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y_pred")])
        test_title = "Integration Test of QMSTest"
        temp_dir = tempfile.mkdtemp()
        json_output = os.path.join(temp_dir, "test.json")
        doc_output = os.path.join(temp_dir, "summary.docx")
        test_descriptions = ["first result is greater than 0", "second result is greater than 0"]
        traces = [
            PassTrace(inputs="y_pred", mode="test"),
            QMSTest(
                test_descriptions=test_descriptions,
                test_criterias=[
                    lambda y_pred: y_pred[0][0][0] > 0,  # 1*1 + 1*2 + 1*3 > 0
                    lambda y_pred: y_pred[0][1][0] > 0,  # 1*1 + 2*2 + 3*(-3) > 0
                ],
                test_title=test_title,
                json_output=json_output,
                doc_output=doc_output)
        ]
        estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=traces)

        with patch("fastestimator.trace.io.qms.json.dump") as fake_dump, \
            patch("fastestimator.trace.io.qms._QMSDocx") as fake_qms:

            estimator.test()
            (json_summary, json_fp), _ = fake_dump.call_args

            with self.subTest("check json summary dict"):
                ans = {
                    "title": test_title,
                    "stories": [{
                        "description": test_descriptions[0], "passed": "True"
                    }, {
                        "description": test_descriptions[1], "passed": "False"
                    }]
                }
                self.assertTrue(is_equal(json_summary, ans))

            with self.subTest("check json summary stored path"):
                self.assertTrue(is_equal(json_fp.name, json_output))

            with self.subTest("check call the _QMSDocx correctly"):
                ans = (1, 1)  # (total_pass, total_fail)
                self.assertEqual(fake_qms.call_args[0], ans)


class TestQMSDocx(unittest.TestCase):
    def test_qmsdocx_can_write(self):
        qms_doc = fe.trace.io.qms._QMSDocx(1, 1)
        file_path = os.path.join(tempfile.mkdtemp(), "test.docx")
        qms_doc.save(file_path)

        with self.subTest("output file exist"):
            self.assertTrue(os.path.exists(file_path))

        with self.subTest("output file is not empty"):
            self.assertGreater(os.stat(file_path).st_size, 0)
