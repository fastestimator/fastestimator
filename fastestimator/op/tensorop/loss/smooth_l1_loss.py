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


class SmoothL1Loss(Loss):
    def __init__(self, y_true=None, y_pred=None, inputs=None, outputs=None, mode=None):
        """Calculate Smooth L1 loss

       Args:
           y_true: ground truth label key
           y_pred: prediction label key
           inputs: A tuple or list like: [<y_true>, <y_pred>]
           outputs: Where to store the computed loss value (not required under normal use cases)
           mode: 'train', 'eval', 'test', or None
       """

        inputs = self.validate_loss_inputs(inputs, y_true, y_pred)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):

        true, pred = data
        diff = tf.abs(true - pred)
        lt_one = tf.dtypes.cast(tf.less(diff, 1.0), "float32")

        loss = lt_one * 0.5 * diff**2 + (1 - lt_one) * (diff - 0.5)

        return loss
