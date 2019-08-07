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
import tensorflow as tf
from fastestimator.network.loss import Loss
from fastestimator.util.op import TensorOp


def build(keras_model, loss, optimizer):
    assert isinstance(keras_model, tf.keras.Model), "must provide tf.keras.Model instance as keras model"
    assert isinstance(loss, Loss)
    keras_model.loss = loss
    if isinstance(optimizer, str):
        optimizer_fn = {
            'adadelta': tf.optimizers.Adadelta,
            'adagrad': tf.optimizers.Adagrad,
            'adam': tf.optimizers.Adam,
            'adamax': tf.optimizers.Adamax,
            'nadam': tf.optimizers.Nadam,
            'rmsprop': tf.optimizers.RMSprop,
            'sgd': tf.optimizers.SGD
        }
        keras_model.optimizer = optimizer_fn[optimizer]()
    else:
        assert isinstance(optimizer,
                          tf.optimizers.Optimizer), "must provide tf.optimizer.Optimizer instance as optimizer"
        keras_model.optimizer = optimizer
    keras_model.fe_compiled = True
    return keras_model


class ModelOp(TensorOp):
    def __init__(self, model, inputs=None, outputs=None, mode=None):
        super(ModelOp, self).__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        assert isinstance(
            self.model, tf.keras.Model
        ) and self.model.fe_compiled is True, "must prepare your the keras model before use in ModelOp"

    def forward(self, data, state):
        data = self.model(data, training=state['mode'] == "train")
        return data
