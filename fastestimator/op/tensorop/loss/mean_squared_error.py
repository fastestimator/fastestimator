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
from fastestimator.op.tensorop.loss import Loss


class MeanSquaredError(Loss):
    """Calculate mean squared error loss, the rest of the keyword argument will be passed to
    tf.losses.MeanSquaredError

    Args:
        y_true: ground truth label key
        y_pred: prediction label key
        inputs: A tuple or list like: [<y_true>, <y_pred>]
        outputs: Where to store the computed loss value (not required under normal use cases)
        mode: 'train', 'eval', 'test', or None
        kwargs: Arguments to be passed along to the tf.losses constructor
    """
    def __init__(self, y_true=None, y_pred=None, inputs=None, outputs=None, mode=None, **kwargs):

        if 'reduction' in kwargs:
            raise KeyError("parameter 'reduction' not allowed")
        inputs = self.validate_loss_inputs(inputs, y_true, y_pred)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = tf.losses.MeanSquaredError(reduction='none', **kwargs)

    def forward(self, data, state):
        true, pred = data
        return self.loss_obj(true, pred)
