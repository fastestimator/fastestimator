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

import tensorflow as tf
import torch

from fastestimator.backend.load_model import load_model
from fastestimator.backend.save_model import save_model
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class RestoreWizard(Trace):
    """A trace that can backup and load your entire training status.

    System includes model weights, optimizer state, global step and epoch index.

    Args:
        directory: Directory to save and load system.
        frequency: Saving frequency in epoch(s).
    """
    def __init__(self, directory: str, frequency: int = 1) -> None:
        super().__init__(mode="train")
        self.directory = directory
        self.frequency = frequency
        self.model_extension = {"tf": "h5", "torch": "pt"}
        self.optimizer_extension = {"tf": "pkl", "torch": "pt"}
        self.system_file = "system.json"

    def on_begin(self, data: Data) -> None:
        if not os.path.exists(self.directory) or not os.listdir(self.directory):
            print("FastEstimator-RestoreWizard: Backing up in {}".format(self.directory))
        else:
            self._scan_files()
            self._load_files()
            data.write_with_log("epoch", self.system.epoch_idx)
            print("FastEstimator-RestoreWizard: Restoring from {}, resume training".format(self.directory))

    def _load_files(self) -> None:
        """Restore from files.
        """
        system_path = os.path.join(self.directory, self.system_file)
        self.system.load_state(json_path=system_path)
        for model in self.system.network.models:
            if isinstance(model, tf.keras.Model):
                framework = "tf"
            elif isinstance(model, torch.nn.Module):
                framework = "torch"
            else:
                raise ValueError("Unknown model type {}".format(type(model)))
            weights_path = os.path.join(self.directory,
                                        "{}.{}".format(model.model_name, self.model_extension[framework]))
            load_model(model, weights_path=weights_path, load_optimizer=True)

    def _scan_files(self) -> None:
        """Scan necessary files to load.
        """
        system_path = os.path.join(self.directory, self.system_file)
        assert os.path.exists(system_path), "cannot find system file at {}".format(system_path)
        for model in self.system.network.models:
            if isinstance(model, tf.keras.Model):
                framework = "tf"
            elif isinstance(model, torch.nn.Module):
                framework = "torch"
            else:
                raise ValueError("Unknown model type {}".format(type(model)))
            weights_path = os.path.join(self.directory,
                                        "{}.{}".format(model.model_name, self.model_extension[framework]))
            assert os.path.exists(weights_path), "cannot find model weights file at {}".format(weights_path)
            optimizer_path = os.path.join(self.directory,
                                          "{}_opt.{}".format(model.model_name, self.optimizer_extension[framework]))
            assert os.path.exists(optimizer_path), "cannot find model optimizer file at {}".format(optimizer_path)

    def on_epoch_end(self, data: Data) -> None:
        if self.system.epoch_idx % self.frequency == 0:
            # Save all models and optimizer state
            for model in self.system.network.models:
                save_model(model, save_dir=self.directory, save_optimizer=True)
            # Save system state
            self.system.save_state(json_path=os.path.join(self.directory, self.system_file))
            print("FastEstimator-RestoreWizard: Saved milestones to {}".format(self.directory))
