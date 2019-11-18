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
from tensorflow.keras import layers


class UncertaintyWeightedLoss(layers.Layer):
    """Creates Uncertainty weighted loss layer https://arxiv.org/abs/1705.07115

    """
    def __init__(self, num_losses):
        super(UncertaintyWeightedLoss, self).__init__()
        self.num_losses = num_losses
        self.log_vars = []

    def build(self, input_shape):
        for idx in range(self.num_losses):
            self.log_vars.append(self.add_weight(shape=(), initializer='zeros', trainable=True))

    def call(self, loss_lists):
        loss = 0
        for idx in range(self.num_losses):
            loss += tf.exp(-self.log_vars[idx]) * loss_lists[idx] + self.log_vars[idx]
        return loss
