# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
from typing import Union, Iterable, Callable, TypeVar, List, Dict, Any

import tensorflow as tf
import torch

from fastestimator.backend.feed_forward import feed_forward
from fastestimator.op.op import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class ModelOp(TensorOp):
    """This class represents the Model operator that defines String keys for storing batch data and predictions

    Args:
        model : keras model compiled by fe.build
        inputs : String key of input training data. Defaults to None.
        outputs : String key of predictions. Defaults to None.
        mode : 'train' or 'eval'. Defaults to None.
        track_input : If 'true' it tracks the gradients with respect to inputs. Defaults to False.  # TODO - bring back
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 trainable: bool = True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        assert hasattr(model, "fe_compiled"), "must use fe.build to compile the model before use"
        self.model = model
        self.trainable = trainable

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        training = state['mode'] == "train" and self.trainable
        data = feed_forward(self.model, data, training=training)
        return data
