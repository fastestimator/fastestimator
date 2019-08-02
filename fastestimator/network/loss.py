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

class Loss:
    def __init__(self):
        pass

    def calculate_loss(self, batch, prediction):
        """this is the function that calculates the loss given the batch data
        
        Args:
            batch (dict): batch data before forward operation
            prediction(dict): prediction data after forward operation
        
        Returns:
            loss (scalar): scalar loss for the model update
        """
        loss = None
        return loss


class SparseCategoricalCrossentropy(Loss):
    def __init__(self, true_key, pred_key, **kwargs):
        """Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed to tf.losses.SparseCategoricalCrossentropy
        
        Args:
            true_key (str): the key of ground truth label in batch data
            pred_key (str): the key of predicted label in batch data
        """
        self.true_key = true_key
        self.pred_key = pred_key
        self.loss_obj = tf.losses.SparseCategoricalCrossentropy(**kwargs)

    def calculate_loss(self, batch, prediction):
        loss = self.loss_obj(batch[self.true_key], prediction[self.pred_key])
        return loss


