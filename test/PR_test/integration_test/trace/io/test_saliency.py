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

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.trace.io import ImageSaver
from fastestimator.trace.xai import Saliency


class TestSaliency(unittest.TestCase):
    """ This test has dependency on:
    * fe.trace.ImageSaver
    """
    def test_saliency(self):
        label_mapping = {
            'zero': 0,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9
        }

        batch_size = 32

        train_data, _ = mnist.load_data()
        test_data = train_data.split([i for i in range(10)])
        pipeline = fe.Pipeline(test_data=test_data,
                               batch_size=batch_size,
                               ops=[ExpandDims(inputs="x", outputs="x", axis=-1), Minmax(inputs="x", outputs="x")])

        weight_path = os.path.abspath(os.path.join(__file__, "..", "resources", "lenet_mnist_tf.h5"))

        model = fe.build(model_fn=lambda: LeNet(input_shape=(28, 28, 1)), optimizer_fn="adam", weights_path=weight_path)
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y_pred")])

        save_dir = tempfile.mkdtemp()
        traces = [
            Saliency(model=model,
                     model_inputs="x",
                     class_key="y",
                     model_outputs="y_pred",
                     samples=5,
                     label_mapping=label_mapping,
                     smoothing=0,
                     integrating=0),
            ImageSaver(inputs="saliency", save_dir=save_dir)
        ]

        estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=5, traces=traces, log_steps=1000)
        estimator.test()
        # we are no longer doing a image pixel-wise match, because the test is extremely brittle, and works differently
        # in different hardwares.
