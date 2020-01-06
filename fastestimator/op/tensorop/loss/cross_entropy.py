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
from fastestimator.backend.cross_entropy import cross_entropy
from fastestimator.backend.reduce_loss import reduce_loss
from fastestimator.op.op import TensorOp


class CrossEntropy(TensorOp):
    """Calculate Element-Wise CrossEntropy(binary, categorical or sparse categorical)

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>]
        outputs: key to store the computed loss value
        mode: 'train', 'eval' or None
        apply_softmax: whether to apply softmax to y_pred. Defaults to False.
        average_loss: whether to average the element-wise loss after the Loss Op
    """
    def __init__(self, inputs=None, outputs=None, mode=None, apply_softmax=False, average_loss=True):
        self.apply_softmax = apply_softmax
        self.average_loss = average_loss
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        y_pred, y_true = data
        loss = cross_entropy(y_pred, y_true, apply_softmax=self.apply_softmax)
        if self.average_loss:
            loss = reduce_loss(loss)
        return loss
