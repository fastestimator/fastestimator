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
import shutil
from typing import List, Union

import fastestimator as fe
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list


@traceable()
class RestoreWizard(Trace):
    """A trace that can backup and load your entire training status.

    Args:
        directory: Directory to save and load the training status.
        frequency: Saving frequency in epoch(s).
    """
    def __init__(self, directory: str, frequency: int = 1) -> None:
        super().__init__(inputs="*", mode="train")  # inputs to cause this trace to sort to the end of the list
        self.directory = os.path.abspath(os.path.normpath(directory))
        self.frequency = frequency
        # For robust saving, we need to create 2 different directories and have a key file to switch between them
        self.dirs = [os.path.join(self.directory, 'A'), os.path.join(self.directory, 'B')]
        self.key_path = os.path.join(self.directory, 'key.txt')
        self.dir_idx = 0

    def on_begin(self, data: Data) -> None:
        if fe.fe_deterministic_seed is not None:
            raise RuntimeError("You cannot use RestoreWizard while in deterministic training mode since a restored" +
                               " training can't guarantee that all prngs will be reset to exactly the same position")
        if not self.should_restore():
            self._cleanup(self.dirs)  # Remove any partially completed checkpoints
            print("FastEstimator-RestoreWizard: Backing up to {}".format(self.directory))
        else:
            self._load_key()
            directory = self.dirs[self.dir_idx]
            self.system.load_state(directory)
            data.write_with_log("epoch", self.system.epoch_idx)
            print("FastEstimator-RestoreWizard: Restoring from {}, resume training".format(directory))
            self.dir_idx = int(not self.dir_idx)  # Flip the idx so that next save goes to other dir
            self._cleanup(self.dirs[self.dir_idx])  # Clean out the other dir in case it had a partial save

    def on_epoch_end(self, data: Data) -> None:
        if self.system.epoch_idx % self.frequency == 0:
            directory = self.dirs[self.dir_idx]
            self.system.save_state(directory)
            self._write_key()
            # Everything after this is free to die without causing problems with restore
            self.dir_idx = int(not self.dir_idx)
            self._cleanup(self.dirs[self.dir_idx])
            print("FastEstimator-RestoreWizard: Saved milestones to {}".format(directory))

    def should_restore(self) -> bool:
        """Whether a restore will be performed.

        Returns:
            True iff the wizard will perform a restore.
        """
        return os.path.exists(self.directory) and os.path.exists(self.key_path)

    def _load_key(self) -> None:
        """Set the dir_idx based on the key last saved by the restore wizard.

        Raises:
            ValueError: If the key file has been modified.
        """
        with open(self.key_path, 'r') as key_file:
            key = key_file.readline()
        if key not in ('A', 'B'):
            raise ValueError("RestoreWizard encountered an invalid key file at {}. Either delete it to restart, or undo"
                             " whatever manual changes were made to the file.".format(self.key_path))
        self.dir_idx = 0 if key == 'A' else 1

    def _write_key(self) -> None:
        """Generate a new key file and then atomically replace the old key file.
        """
        sub_dir = self.dirs[self.dir_idx]
        new_key_path = os.path.join(sub_dir, 'key.txt')
        with open(new_key_path, 'w') as new_key_file:
            new_key_file.write("B" if self.dir_idx else "A")
        os.replace(new_key_path, self.key_path)  # This operation is atomic per POSIX requirements

    @staticmethod
    def _cleanup(paths: Union[str, List[str]]) -> None:
        """Delete stale directories if they exist.

        Args:
            paths: Which directories to delete.
        """
        paths = to_list(paths)
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
