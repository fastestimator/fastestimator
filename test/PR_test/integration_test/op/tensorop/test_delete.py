# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
import torch
import torch.nn as nn
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import fastestimator as fe
from fastestimator.op.tensorop import Delete, LambdaOp
from fastestimator.op.tensorop.model import ModelOp


def _TFModel():
    inputs = Input((1, 1))
    return Model(inputs=inputs, outputs=inputs)


class _TorchModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _new_key_network(model):
    return fe.Network(ops=[
        ModelOp(inputs="x", outputs="y_pred", model=model),
        LambdaOp(inputs="x", outputs="x", fn=lambda x: x + 1),
        Delete(keys="y_pred")
    ])


def _old_key_network(model):
    return fe.Network(ops=[
        ModelOp(inputs="x", outputs="y_pred", model=model),
        LambdaOp(inputs="x", outputs="x", fn=lambda x: x + 1),
        Delete(keys="x")
    ])


def _torch_model():
    return fe.build(
        model_fn=_TorchModel,
        optimizer_fn="adam",
    )


def _tf_model():
    return fe.build(
        model_fn=_TFModel,
        optimizer_fn="adam",
    )


def _batch():
    return {"x": np.ones((4, 1), dtype=np.float32), "y": np.zeros((4, 1), dtype=np.uint8)}


class TestSlicer(unittest.TestCase):
    def test_delete_new_key_transform_tf(self):
        result = _new_key_network(model=_tf_model()).transform(data=_batch(), mode="test")
        self.assertIn("x", result)
        np.testing.assert_array_almost_equal(result['x'], np.array([[2], [2], [2], [2]]))
        self.assertNotIn("y_pred", result)

    def test_delete_old_key_transform_tf(self):
        result = _old_key_network(model=_tf_model()).transform(data=_batch(), mode="test")
        # X will still be in the response, but it's value should be the old input value rather than the updated value
        np.testing.assert_array_almost_equal(result["x"], np.array([[1], [1], [1], [1]]))
        self.assertIn("y_pred", result)

    def test_delete_new_key_transform_torch(self):
        result = _new_key_network(model=_torch_model()).transform(data=_batch(), mode="test")
        self.assertIn("x", result.maps[0])
        self.assertNotIn("y_pred", result.maps[0])

    def test_delete_old_key_transform_torch(self):
        result = _old_key_network(model=_torch_model()).transform(data=_batch(), mode="test")
        self.assertNotIn("x", result.maps[0])
        self.assertIn("y_pred", result.maps[0])
