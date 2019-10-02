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
from fastestimator.op import TensorOp


class Scheduler:
    def __init__(self, epoch_dict):
        self.epoch_dict = epoch_dict
        self.value = None
        self._verify_inputs()

    def _verify_inputs(self):
        assert isinstance(self.epoch_dict, dict), "must provide dictionary as epoch_dict"
        self.keys = sorted(self.epoch_dict)
        sample_content = self._get_sample_content()
        for key in self.keys:
            assert isinstance(key, int), "found non-integer key: {}".format(key)
            assert key >= 0, "found negative key: {}".format(key)
            if isinstance(sample_content, TensorOp) and self.epoch_dict[key]:
                assert self.mode == self.epoch_dict[key].mode, "schedule contents must have same mode"

    def get_sequential_value(self, epoch):
        if epoch in self.keys:
            self.value = self.epoch_dict[epoch]
        return self.value

    def get_current_value(self, epoch):
        if epoch in self.keys:
            value = self.epoch_dict[epoch]
        else:
            last_key = self._get_last_key(epoch)
            if last_key is None:
                value = None
            else:
                value = self.epoch_dict[last_key]
        return value

    def _get_last_key(self, epoch):
        last_key = None
        for key in self.keys:
            if key > epoch:
                break
            last_key = key
        return last_key

    def _get_sample_content(self):
        sample_content = None
        for value in self.epoch_dict.values():
            if value is not None:
                sample_content = value
                break
        if isinstance(sample_content, TensorOp):
            self.mode = sample_content.mode
        assert sample_content is not None, "At least one value in a scheduler dict must be non-None"
        return sample_content
