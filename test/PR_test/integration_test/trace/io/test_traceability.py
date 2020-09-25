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
import shutil
import tempfile
import unittest
from typing import Union

import pydot
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet as PyLeNet
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import Traceability
from fastestimator.trace.metric import Accuracy
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import FeInputSpec


def _lacks_graphviz():
    try:
        pydot.Dot.create(pydot.Dot())
    except OSError:
        return True
    return False


def _build_estimator(model: Union[tf.keras.Model, torch.nn.Module], trace: Traceability, axis: int = -1):
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    batch_size = 32
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x", axis=axis), Minmax(inputs="x", outputs="x")])
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3)),
        trace
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=1,
                             traces=traces,
                             max_train_steps_per_epoch=1,
                             max_eval_steps_per_epoch=None)
    fake_data = tf.ones(shape=(batch_size, 28, 28, 1)) if axis == -1 else torch.ones(size=(batch_size, 1, 28, 28))
    model.fe_input_spec = FeInputSpec(fake_data, model)
    return estimator


class TestTraceability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.join(tempfile.gettempdir(), "FEUnitTestReports")
        cls.tf_dir = os.path.join(cls.root_dir, "TF")
        cls.torch_dir = os.path.join(cls.root_dir, "Torch")

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_tf_traceability(self):
        if os.path.exists(self.tf_dir) and os.path.isdir(self.tf_dir):
            shutil.rmtree(self.tf_dir)

        trace = Traceability(save_path=self.tf_dir)
        est = _build_estimator(fe.build(model_fn=LeNet, optimizer_fn="adam", model_name='tfLeNet'), trace)

        trace.system = est.system
        trace.system.epoch_idx = 1
        trace.system.summary.name = "TF Test"

        trace.on_begin(Data())
        trace.on_end(Data())

        crawler = os.walk(self.tf_dir)
        root = next(crawler)
        self.assertIn('resources', root[1], "A resources subdirectory should have been generated")
        self.assertIn('tf_test.tex', root[2], "The tex file should have been generated")
        # Might be a pdf and/or a .ds_store file depending on system, but shouldn't be more than that
        self.assertLessEqual(len(root[2]), 3, "Extra files should not have been generated")
        figs = next(crawler)
        self.assertIn('tf_test_tfLeNet.pdf', figs[2], "A figure for the model should have been generated")
        self.assertIn('tf_test_logs.png', figs[2], "A log image should have been generated")
        self.assertIn('tf_test.txt', figs[2], "A raw log file should have been generated")

    @unittest.skipIf(_lacks_graphviz(), "The machine does not have GraphViz installed")
    def test_torch_traceability(self):
        if os.path.exists(self.torch_dir) and os.path.isdir(self.torch_dir):
            shutil.rmtree(self.torch_dir)

        trace = Traceability(save_path=self.torch_dir)
        est = _build_estimator(fe.build(model_fn=PyLeNet, optimizer_fn="adam", model_name='torchLeNet'), trace, axis=0)

        trace.system = est.system
        trace.system.epoch_idx = 1
        trace.system.summary.name = "Torch Test"

        trace.on_begin(Data())
        trace.on_end(Data())

        crawler = os.walk(self.torch_dir)
        root = next(crawler)
        self.assertIn('resources', root[1], "A figures subdirectory should have been generated")
        self.assertIn('torch_test.tex', root[2], "The tex file should have been generated")
        # Might be a pdf and/or a .ds_store file depending on system, but shouldn't be more than that
        self.assertLessEqual(len(root[2]), 3, "Extra files should not have been generated")
        figs = next(crawler)
        self.assertIn('torch_test_torchLeNet.pdf', figs[2], "A figure for the model should have been generated")
        self.assertIn('torch_test_logs.png', figs[2], "A log image should have been generated")
        self.assertIn('torch_test.txt', figs[2], "A raw log file should have been generated")
