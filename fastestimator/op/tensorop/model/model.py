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
from fastestimator.op import TensorOp


class ModelOp(TensorOp):
    """This class represents the Model operator that defines String keys for storing batch data and predictions

    Args:
        model : keras model compiled by fe.build
        inputs : String key of input training data. Defaults to None.
        outputs : String key of predictions. Defaults to None.
        mode : 'train' or 'eval'. Defaults to None.
        track_input : If 'true' it tracks the gradients with respect to inputs. Defaults to False.
    """
    def __init__(self, model, inputs=None, outputs=None, mode=None, trainable=True, track_input=False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        assert hasattr(model, "fe_compiled"), "must use fe.build to compile the model before use"
        self.model = model
        self.trainable = trainable
        self.track_input = track_input

    def forward(self, data, state):
        training = state['mode'] == "train" and self.trainable
        if self.track_input and training:
            tape = state['tape']
            tape.watch(data)
        data = self.model(data, training=training)
        return data
