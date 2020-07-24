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
import queue
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
        max_to_keep: Maximum number of latest saved files to keep.
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
        self.max_to_keep = max_to_keep
        if max_to_keep:
            self.file_queue = queue.Queue()

    def on_epoch_end(self, data: Data) -> None:
        # No model will be saved when save_dir is None, which makes smoke test easier.
        if self.save_dir and self.system.epoch_idx % self.frequency == 0:
            model_name = "{}_epoch_{}".format(self.model.model_name, self.system.epoch_idx)
            model_path = save_model(self.model, self.save_dir, model_name)
            print("FastEstimator-ModelSaver: Saved model to {}".format(model_path))

            if self.max_to_keep:
                if self.file_queue.qsize() == self.max_to_keep:
                    removed_name = self.file_queue.get()
                    if isinstance(self.model, tf.keras.Model):
                        removed_name = "{}.h5".format(removed_name)
                    elif isinstance(self.model, torch.nn.Module):
                        removed_name = "{}.pt".format(removed_name)
                    else:
                        raise ValueError("Unrecognized model instance {}".format(type(model)))
                    os.remove(os.path.join(self.save_dir, removed_name))
                    print("FastEstimator-ModelSaver: Removed model {} due to file number exceeding max_to_keep".format(
                        model_path))
                self.file_queue.put(model_name)
