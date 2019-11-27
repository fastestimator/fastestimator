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
from fastestimator.layers.uncertainty_weighted_loss import UncertaintyWeightedLoss
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model


def UncertaintyLossNet(num_losses=2):
    """Creates Uncertainty weighted loss model https://arxiv.org/abs/1705.07115

    Args:
        num_losses: the number of losses

    Returns:
        'Model' object: Model that produce the uncertainty weighted loss
    """
    assert num_losses > 1, "number of losses must be greater than 1 for uncertainty weighted loss"
    input_layers = []
    for _ in range(num_losses):
        input_layers.append(layers.Input(shape=[]))
    loss = UncertaintyWeightedLoss(num_losses=num_losses)(input_layers)
    model = Model(inputs=input_layers, outputs=loss)
    return model
