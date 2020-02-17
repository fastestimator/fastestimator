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

from fastestimator.op import TensorOp


class AdversarialSample(TensorOp):
    """ This class is to be used to train the model more robust against adversarial attacks (
    https://arxiv.org/abs/1412.6572)

    Args:
        inputs (str): key of the input to be attacked
        loss (str): key of the loss value to use in the attack - mutually exclusive with gradients
        gradients (str): key of the gradients to use in the attack - mutually exclusive with loss
        outputs (str): key to store the mixed-up input
        mode (str): what mode to execute in.
        epsilon (float): epsilon value to perturb the input to create adversarial examples
        clip_low (float): a minimum value to clip the output by (defaults to min value of data)
        clip_high (float): a maximum value to clip the output by (defaults to max value of data)
    """
    def __init__(self,
                 inputs,
                 loss=None,
                 gradients=None,
                 outputs=None,
                 mode=None,
                 epsilon=0.01,
                 clip_low=None,
                 clip_high=None):
        assert (loss or gradients) is not None and not (loss and gradients) is not None, \
            "AdversarialSample requires either a loss key or a gradient key, but not both"
        self.loss_mode = loss is not None
        super().__init__(inputs=[loss or gradients, inputs], outputs=outputs, mode=mode)
        self.epsilon = epsilon
        self.clip_low = clip_low
        self.clip_high = clip_high

    def forward(self, data, state):
        """ Forward method to perform mixup batch augmentation

        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.

        Returns:
            Adversarial example created from perturbing the input data
        """
        gradients, data = data
        tape = state['tape']
        with tape.stop_recording():
            if self.loss_mode:
                gradients = tape.gradient(gradients, data)
            adverse_data = tf.clip_by_value(data + self.epsilon * tf.sign(gradients),
                                            self.clip_low or tf.reduce_min(data),
                                            self.clip_high or tf.reduce_max(data))
        return adverse_data
