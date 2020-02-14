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
from typing import Optional, Union

import tensorflow as tf
from torch.utils.data import DataLoader

from fastestimator.util.util import get_num_devices


class System:
    mode: str  # What is the current execution mode of the estimator ('train', 'eval', 'test')
    global_step: int  # How many training steps have elapsed
    num_devices: int  # How many GPUs are available for training
    log_steps: Optional[int]  # Log every n steps (0 to disable train logging, None to disable all logging)
    epochs: int  # How many total epochs training is expected to run for
    epoch_idx: int  # The current epoch index for the training (starting from 0)
    batch_idx: int  # The current batch index within an epoch (starting from 0)
    stop_training: bool  # A flag to signal that training should abort before 'epochs' has been reached
    loader: Union[None, DataLoader, tf.data.Dataset]  # A reference to the object loading data this epoch
    network: Optional[object]  # A reference to the network being used this epoch  # TODO - circular reference

    def __init__(self,
                 mode: str = "train",
                 num_devices: int = get_num_devices(),
                 log_steps: Optional[int] = None,
                 epochs: int = 0):
        self.mode = mode
        self.global_step = 0
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.epochs = epochs
        self.epoch_idx = 0
        self.batch_idx = 0
        self.stop_training = False

    def update_global_step(self):
        self.global_step += 1

    def reset(self):
        self.mode = "train"
        self.global_step = 0
        self.epoch_idx = 0
        self.batch_idx = 0
        self.stop_training = False
        self.loader = None
        self.network = None
