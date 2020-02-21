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
from typing import Any, Optional

from fastestimator.network import BaseNetwork
from fastestimator.util.util import get_num_devices
from fastestimator.summary.summary import Summary


class System:
    mode: str  # What is the current execution mode of the estimator ('train', 'eval', 'test')
    global_step: int  # How many training steps have elapsed
    num_devices: int  # How many GPUs are available for training
    log_steps: Optional[int]  # Log every n steps (0 to disable train logging, None to disable all logging)
    total_epochs: int  # How many epochs training is expected to run for
    epoch_idx: int  # The current epoch index for the training (starting from 0)
    batch_idx: int  # The current batch index within an epoch (starting from 0)
    stop_training: bool  # A flag to signal that training should abort
    network: BaseNetwork  # A reference to the network being used this epoch
    max_steps_per_epoch: Optional[int]  # Training epoch will complete after n steps even if loader is not yet exhausted
    summary: Summary  # An object to write experiment results to

    def __init__(self,
                 network: BaseNetwork,
                 mode: str = "train",
                 num_devices: int = get_num_devices(),
                 log_steps: Optional[int] = None,
                 total_epochs: int = 0,
                 max_steps_per_epoch: Optional[int] = None):

        self.network = network
        self.mode = mode
        self.global_step = 0
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.epoch_idx = 0
        self.batch_idx = 0
        self.max_steps_per_epoch = max_steps_per_epoch
        self.stop_training = False
        self.summary = Summary(None)

    def update_global_step(self):
        self.global_step += 1

    def reset(self, summary_name: Optional[str] = None):
        self.mode = "train"
        self.global_step = 0
        self.epoch_idx = 0
        self.batch_idx = 0
        self.stop_training = False
        self.summary = Summary(summary_name)

    def reset_for_test(self, summary_name: Optional[str] = None):
        self.mode = "test"
        if not self.stop_training:
            self.epoch_idx = self.total_epochs - 1
        self.stop_training = False
        self.summary.name = summary_name or self.summary.name  # Keep old experiment name if new one not provided
        self.summary.history.pop('test', None)

    def write_summary(self, key: str, value: Any):
        if self.summary:
            self.summary.history[self.mode][key][self.global_step] = value
