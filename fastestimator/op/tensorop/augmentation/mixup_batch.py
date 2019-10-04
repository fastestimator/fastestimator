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
import tensorflow_probability as tfp

from fastestimator.op import TensorOp


class MixUpBatch(TensorOp):
    """ This class should be used in conjunction with MixUpLoss to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412)

    Args:
        inputs: key of the input to be mixed up
        outputs: key to store the mixed-up input
        mode: what mode to execute in. Probably 'train'
        alpha: the alpha value defining the beta distribution to be drawn from during training
    """
    def __init__(self, inputs=None, outputs=None, mode=None, alpha=1.0):
        assert alpha > 0, "Mixup alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = tf.constant(alpha)
        self.beta = tfp.distributions.Beta(alpha, alpha)

    def forward(self, data, state):
        """ Forward method to perform mixup batch augmentation

        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.

        Returns:
            Mixed-up batch data
        """
        iterdata = data if isinstance(data, list) else list(data) if isinstance(data, tuple) else [data]
        lam = self.beta.sample()
        # Could do random mix-up using tf.gather() on a shuffled index list, but batches are already randomly ordered,
        # so just need to roll by 1 to get a random combination of inputs. This also allows MixUpLoss to easily compute
        # the corresponding Y values
        mix = [lam * dat + (1.0 - lam) * tf.roll(dat, shift=1, axis=0) for dat in iterdata]
        return mix + [lam]
