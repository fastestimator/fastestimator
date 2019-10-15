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


class CutMixBatch(TensorOp):
    """ This class should be used in conjunction with MixUpLoss to perform CutMix training, which helps to reduce
    over-fitting, perform object detection, and against adversarial attacks (https://arxiv.org/pdf/1905.04899.pdf)

    Args:
        inputs: key of the input to be cut-mixed
        outputs: key to store the cut-mixed input
        mode: what mode to execute in. Probably 'train'
        alpha: the alpha value defining the beta distribution to be drawn from during training
    """
    def __init__(self, inputs=None, outputs=None, mode=None, alpha=1.0):
        assert alpha > 0, "Mixup alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = tf.constant(alpha)
        self.beta = tfp.distributions.Beta(alpha, alpha)
        self.uniform = tfp.distributions.Uniform()

    def forward(self, data, state):
        """ Forward method to perform cutmix batch augmentation

        Args:
            data: Batch data to be augmented (batch X height X width X channel)
            state: Information about the current execution context.

        Returns:
            Cut-Mixed batch data
        """
        _, height, width, _ = data.shape
        lam = self.beta.sample()
        rx = width * self.uniform.sample()
        ry = height * self.uniform.sample()
        rw = width * tf.sqrt(1 - lam)
        rh = height * tf.sqrt(1 - lam)
        x1 = tf.dtypes.cast(tf.round(tf.maximum(rx - rw / 2, 0)), tf.int32)
        x2 = tf.dtypes.cast(tf.round(tf.minimum(rx + rw / 2, width)), tf.int32)
        y1 = tf.dtypes.cast(tf.round(tf.maximum(ry - rh / 2, 0)), tf.int32)
        y2 = tf.dtypes.cast(tf.round(tf.minimum(ry + rh / 2, height)), tf.int32)

        patches = tf.roll(data, shift=1, axis=0)[:, y1:y2, x1:x2, :] - data[:, y1:y2, x1:x2, :]
        patches = tf.pad(patches, [[0, 0], [y1, height - y2], [x1, width - x2], [0, 0]],
                         mode="CONSTANT",
                         constant_values=0)

        lam = tf.dtypes.cast(1.0 - (x2 - x1) * (y2 - y1) / (width * height), tf.float32)
        return data + patches, lam
