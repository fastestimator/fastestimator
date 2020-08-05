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
import os
from collections import deque
from typing import Optional, Union

import tensorflow as tf
import torch

from fastestimator.backend.save_model import save_model
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class ModelSaver(Trace):
    """Save model weights based on epoch frequency during training.

    Args:
        model: A model instance compiled with fe.build.
        save_dir: Folder path into which to save the `model`.
        frequency: Model saving frequency in epoch(s).
        max_to_keep: Maximum number of latest saved files to keep. If 0 or None, all models will be saved.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 save_dir: str,
                 frequency: int = 1,
                 max_to_keep: Optional[int] = None) -> None:
        super().__init__(mode="train")
        self.model = model
        self.save_dir = save_dir
        self.frequency = frequency
        if max_to_keep is not None and max_to_keep < 0:
            raise ValueError(f"max_to_keep should be a non-negative integer, but got {max_to_keep}")
        self.file_queue = deque([None] * (max_to_keep or 0), maxlen=max_to_keep or 0)

    def on_epoch_end(self, data: Data) -> None:
        # No model will be saved when save_dir is None, which makes smoke test easier.
        if self.save_dir and self.system.epoch_idx % self.frequency == 0:
            model_name = "{}_epoch_{}".format(self.model.model_name, self.system.epoch_idx)
            model_path = save_model(self.model, self.save_dir, model_name)
            print("FastEstimator-ModelSaver: Saved model to {}".format(model_path))
            rm_path = self.file_queue[self.file_queue.maxlen - 1] if self.file_queue.maxlen else None
            if rm_path:
                os.remove(rm_path)
                print("FastEstimator-ModelSaver: Removed model {} due to file number exceeding max_to_keep".format(
                    rm_path))
            self.file_queue.appendleft(model_path)
