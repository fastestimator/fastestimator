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

class Pipeline:
    def __init__(self, train_data, eval_data=None, batch_size=None, ops=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.ops = ops
        self._initial_check()

    def get_iterator(self, mode):
        ds_iter = None
        if mode == "train":
            ds_iter = self.train_data
        elif mode == "eval":
            ds_iter = self.eval_data
        return ds_iter
    
    def get_batch_size(self, epoch_idx):
        #update this function later for scheduling batch size
        return self.batch_size
    
    def _initial_check(self):
        if isinstance(self.train_data, torch.utils.data.DataLoader):
            self.batch_size = self.train_data.batch_size
        else:
            assert self.batch_size, "must provide batch size in fe.Pipeline"
        