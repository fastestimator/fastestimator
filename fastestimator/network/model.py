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
import types

import tensorflow as tf

from fastestimator.network.loss import Loss
from fastestimator.util.op import TensorOp


class Bundle:
    def __init__(self, model_def, loss_name, optimizer, name):
        assert isinstance(model_def, types.FunctionType), "must provide function definition or lambda function as model_def"
        assert isinstance(loss_name, str), "loss must be a string key of loss tensor"
        self.model_def = model_def
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.name = name
        self._check_optimizer()

    def _check_optimizer(self):
        if isinstance(self.optimizer, str):
            optimizer_fn = {
                'adadelta': tf.optimizers.Adadelta,
                'adagrad': tf.optimizers.Adagrad,
                'adam': tf.optimizers.Adam,
                'adamax': tf.optimizers.Adamax,
                'nadam': tf.optimizers.Nadam,
                'rmsprop': tf.optimizers.RMSprop,
                'sgd': tf.optimizers.SGD
            }
            self.optimizer = optimizer_fn[self.optimizer]()
        else:
            assert isinstance(self.optimizer, tf.optimizers.Optimizer), "must provide tf.optimizer.Optimizer instance as optimizer"


class ModelOp(TensorOp):
    def __init__(self, bundle, inputs=None, outputs=None, mode=None, track_input=False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        assert isinstance(bundle, Bundle), "must provide Bundle in as input"
        self.bundle = bundle
        self.track_input = track_input

    def forward(self, data, state):
        training = state['mode'] == "train"
        if self.track_input and training:
            tape = state['tape']
            tape.watch(data)
        data = self.bundle.model(data, training=training)
        return data
