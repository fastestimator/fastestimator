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
from fastestimator.util.util import get_num_devices


class System:
    def __init__(self,
                 mode: str = "train",
                 global_step: int = 0,
                 num_devices: int = get_num_devices(),
                 log_steps: Optional[int] = None,
                 epochs: int = 0,
                 epoch_idx: int = 0,
                 batch_idx: int = 0):
        self.mode = mode
        self.global_step = global_step
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.epochs = epochs
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.buffer = {}
        self.loader = None
        self.network = None

    def add_buffer(self, key: str, value: Any):
        self.buffer[key] = value

    def clear_buffer(self):
        del self.buffer
        self.buffer = {}

    def read_buffer(self, key: str) -> Any:
        return self.buffer[key]

    def update_global_step(self):
        self.global_step += 1

    def reset(self):
        self.mode = "train"
        self.global_step = 0
        self.epoch_idx = 0
        self.batch_idx = 0
        self.loader = None
        self.network = None
        self.clear_buffer()
