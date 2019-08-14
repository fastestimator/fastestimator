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

from fastestimator.util.op import TensorOp


class Loss(TensorOp):
    """
    A base class for loss operations. It can be used directly to perform value pass-through (see the adversarial
    training showcase for an example of when this is useful)
    """


class BinaryCrossentropy(Loss):
    def __init__(self, true_key, pred_key, **kwargs):
        """Calculate binary cross entropy, the rest of the keyword argument will be passed to
           tf.losses.BinaryCrossentropy

        Args:
            true_key (str): the key of ground truth label in batch data
            pred_key (str): the key of predicted label in batch data
        """
        super().__init__()
        self.true_key = true_key
        self.pred_key = pred_key
        self.loss_obj = tf.losses.BinaryCrossentropy(**kwargs)

    def calculate_loss(self, batch, state):
        loss = self.loss_obj(batch[self.true_key], batch[self.pred_key])
        return loss


class SparseCategoricalCrossentropy(Loss):
    def __init__(self, inputs, outputs=None, mode=None, **kwargs):
        """Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed to
           tf.losses.SparseCategoricalCrossentropy

        Args:
            inputs: A tuple or list like: [<ground truth label key>, <prediction label key>]
            outputs: Where to store the computed loss value (not required under normal use cases)
            mode: 'train', 'eval', 'test', or None
            kwargs: Arguments to be passed along to the tf.losses constructor
        """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2, \
            "SparseCategoricalCrossentropy requires two inputs: <true key>, <predicted key>"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = tf.losses.SparseCategoricalCrossentropy(**kwargs)

    def forward(self, data, state):
        true, pred = data
        return self.loss_obj(true, pred)


class BinaryCrossentropy(Loss):
    def __init__(self, inputs, outputs=None, mode=None, **kwargs):
        """Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed to
                  tf.losses.SparseCategoricalCrossentropy

               Args:
                   inputs: A tuple or list like: [<ground truth label key>, <prediction label key>]
                   outputs: Where to store the computed loss value (not required under normal use cases)
                   mode: 'train', 'eval', 'test', or None
                   kwargs: Arguments to be passed along to the tf.losses constructor
               """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2, \
            "BinaryCrossentropy requires two inputs: <true key>, <predicted key>"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = tf.losses.BinaryCrossentropy(**kwargs)

    def forward(self, data, state):
        true, pred = data
        return self.loss_obj(true, pred)


class MixUpLoss(Loss):
    """
    This class should be used in conjunction with MixUpBatch to perform mix-up training, which helps to reduce
    over-fitting, stabilize GAN training, and harden against adversarial attacks (https://arxiv.org/abs/1710.09412)
    """
    def __init__(self, loss, inputs, outputs=None, mode=None):
        """
        Args:
            loss (func): A loss object (tf.losses) which can be invoked like "loss(true, pred)"
            inputs: Should be a tuple with the lambda key as the first argument, then the true key, then the prediction
        """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 3, \
            "MixUpLoss requires inputs: <lambda key>, <true key>, <predicted key>"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = loss

    def forward(self, data, state):
        lam, true, pred = data
        loss1 = self.loss_obj(true, pred)
        loss2 = self.loss_obj(tf.roll(true, shift=1, axis=0), pred)
        return lam * loss1 + (1.0 - lam) * loss2
