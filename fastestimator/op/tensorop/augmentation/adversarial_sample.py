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
        inputs: key of the input to be mixed up
        outputs: key to store the mixed-up input
        mode: what mode to execute in.
        epsilon: epsilon value to perturb the input to create adversarial examples
    """
    def __init__(self, inputs, outputs=None, mode=None, epsilon=0.1):
        assert len(inputs) == 2, \
            "AdversarialSample requires 2 inputs: a loss value and the input data which lead to the loss"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.epsilon = epsilon

    def forward(self, data, state):
        """ Forward method to perform mixup batch augmentation

        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.

        Returns:
            Adversarial example created from perturbing the input data
        """
        clean_loss, clean_data = data
        tape = state['tape']
        with tape.stop_recording():
            grad_clean = tape.gradient(clean_loss, clean_data)
            adverse_data = tf.clip_by_value(clean_data + self.epsilon * tf.sign(grad_clean),
                                            tf.reduce_min(clean_data),
                                            tf.reduce_max(clean_data))
        return adverse_data
