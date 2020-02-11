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


class Scheduler:
    def get_current_value(self, epoch: int):
        raise NotImplementedError


class RepeatScheduler(Scheduler):
    def __init__(self, repeat_list: list):
        assert isinstance(repeat_list, list), "must provide a list as input of RepeatSchedule"
        self.repeat_list = repeat_list
        self.cycle_length = len(repeat_list)
        assert self.cycle_length > 1, "list length must be greater than 1"

    def get_current_value(self, epoch: int):
        return self.repeat_list[epoch % self.cycle_length]

    def get_items(self):
        return self.repeat_list


class FunctionScheduler(Scheduler):
    def __init__(self, schedule_fun):
        assert hasattr(schedule_fun, '__call__'), "must provide a lambda function with epoch as input"
        self.schedule_fun = schedule_fun

    def get_current_value(self, epoch: int):
        return self.schedule_fun(epoch)


class EpochScheduler(Scheduler):
    def __init__(self, epoch_dict):
        assert isinstance(epoch_dict, dict), "must provide dictionary as epoch_dict"
        self.epoch_dict = epoch_dict
        self.keys = sorted(self.epoch_dict)
        assert 0 in self.epoch_dict, "epoch 0 is missing in dictionary, use None if no op is needed"
        for key in self.keys:
            assert isinstance(key, int), "found non-integer key: {}".format(key)
            assert key >= 0, "found negative key: {}".format(key)

    def get_current_value(self, epoch: int):
        if epoch in self.keys:
            value = self.epoch_dict[epoch]
        else:
            value = self.epoch_dict[self._get_last_key(epoch)]
        return value

    def get_items(self):
        return list(self.epoch_dict.values())

    def _get_last_key(self, epoch: int) -> int:
        last_key = 0
        for key in self.keys:
            if key > epoch:
                break
            last_key = key
        return last_key
