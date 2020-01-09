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
import torch


def torch_to_tf(data):
    # TODO - it might be desirable to replace the collate function of the data loader rather than casting
    #  after-the-fact, but surprisingly tests so far have shown that this doesn't add a noticeable performance penalty
    if isinstance(data, tf.Tensor):
        return data
    if isinstance(data, torch.Tensor):
        return tf.constant(data.numpy(), dtype=tf.float32)
    if isinstance(data, dict):
        result = {}
        for key, val in data.items():
            result[key] = torch_to_tf(val)
        return result
    if isinstance(data, list):
        return [torch_to_tf(val) for val in data]
    if isinstance(data, tuple):
        return tuple([torch_to_tf(val) for val in data])
    if isinstance(data, set):
        return set([torch_to_tf(val) for val in data])
