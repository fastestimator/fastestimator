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
import datetime
import json
from typing import Any, List, Optional, TYPE_CHECKING, Union

import torch

from fastestimator.network import BaseNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler
from fastestimator.summary.summary import Summary
from fastestimator.util.traceability_util import FeSummaryTable

if TYPE_CHECKING:
    from fastestimator.trace.trace import Trace


class System:
    """A class which tracks state information while the fe.Estimator is running.

    This class is intentionally not @traceable.

    Args:
        network: The network instance being used by the current fe.Estimator.
        pipeline: The pipeline instance being used by the current fe.Estimator.
        traces: The traces provided to the current fe.Estimator.
        mode: The current execution mode (or None for warmup).
        num_devices: How many GPUs are available for training.
        log_steps: Log every n steps (0 to disable train logging, None to disable all logging).
        total_epochs: How many epochs training is expected to run for.
        max_train_steps_per_epoch: Whether training epochs will be cut short after N steps (or use None if they will run
            to completion)
        system_config: A description of the initialization parameters defining the associated estimator.

    Attributes:
        mode: What is the current execution mode of the estimator ('train', 'eval', 'test'), None if warmup.
        global_step: How many training steps have elapsed.
        num_devices: How many GPUs are available for training.
        log_steps: Log every n steps (0 to disable train logging, None to disable all logging).
        total_epochs: How many epochs training is expected to run for.
        epoch_idx: The current epoch index for the training (starting from 1).
        batch_idx: The current batch index within an epoch (starting from 1).
        stop_training: A flag to signal that training should abort.
        network: A reference to the network being used.
        pipeline: A reference to the pipeline being used.
        traces: The traces being used.
        max_train_steps_per_epoch: Training will complete after n steps even if loader is not yet exhausted.
        max_eval_steps_per_epoch: Evaluation will complete after n steps even if loader is not yet exhausted.
        summary: An object to write experiment results to.
        experiment_time: A timestamp indicating when this model was trained.
    """

    mode: Optional[str]
    global_step: Optional[int]
    num_devices: int
    log_steps: Optional[int]
    total_epochs: int
    epoch_idx: Optional[int]
    batch_idx: Optional[int]
    stop_training: bool
    network: BaseNetwork
    pipeline: Pipeline
    traces: List[Union['Trace', Scheduler['Trace']]]
    max_train_steps_per_epoch: Optional[int]
    max_eval_steps_per_epoch: Optional[int]
    summary: Summary
    experiment_time: str

    def __init__(self,
                 network: BaseNetwork,
                 pipeline: Pipeline,
                 traces: List[Union['Trace', Scheduler['Trace']]],
                 mode: Optional[str] = None,
                 num_devices: int = torch.cuda.device_count(),
                 log_steps: Optional[int] = None,
                 total_epochs: int = 0,
                 max_train_steps_per_epoch: Optional[int] = None,
                 max_eval_steps_per_epoch: Optional[int] = None,
                 system_config: Optional[List[FeSummaryTable]] = None) -> None:

        self.network = network
        self.pipeline = pipeline
        self.traces = traces
        self.mode = mode
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.batch_idx = None
        self.max_train_steps_per_epoch = max_train_steps_per_epoch
        self.max_eval_steps_per_epoch = max_eval_steps_per_epoch
        self.stop_training = False
        self.summary = Summary(None, system_config)
        self.experiment_time = ""
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize the training state.
        """
        self.global_step = None
        self.epoch_idx = 0

    def load_state(self, json_path) -> None:
        """Load training state.

        Args:
            json_path: The json file path to load from.
        """
        with open(json_path, 'r') as fp:
            state = json.load(fp)
        self.epoch_idx = state["epoch_idx"]
        self.global_step = state["global_step"]

    def save_state(self, json_path) -> None:
        """Load training state.

        Args:
            json_path: The json file path to save to.
        """
        # TODO "summary" and "experiment_time" needs to be saved in the future
        state = {"epoch_idx": self.epoch_idx, "global_step": self.global_step}
        with open(json_path, 'w') as fp:
            json.dump(state, fp, indent=4)

    def update_global_step(self) -> None:
        """Increment the current `global_step`.
        """
        if self.global_step is None:
            self.global_step = 1
        else:
            self.global_step += 1

    def update_batch_idx(self) -> None:
        """Increment the current `batch_idx`.
        """
        if self.batch_idx is None:
            self.batch_idx = 1
        else:
            self.batch_idx += 1

    def reset(self, summary_name: Optional[str] = None, system_config: Optional[str] = None) -> None:
        """Reset the current `System` for a new round of training, including a new `Summary` object.

        Args:
            summary_name: The name of the experiment. The `Summary` object will store information iff name is not None.
            system_config: A description of the initialization parameters defining the associated estimator.
        """
        self.experiment_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = "train"
        self._initialize_state()
        self.batch_idx = None
        self.stop_training = False
        self.summary = Summary(summary_name, system_config)

    def reset_for_test(self, summary_name: Optional[str] = None) -> None:
        """Partially reset the current `System` object for a new round of testing.

        Args:
            summary_name: The name of the experiment. If not provided, the system will re-use the previous summary name.
        """
        self.experiment_time = self.experiment_time or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = "test"
        if not self.stop_training:
            self.epoch_idx = self.total_epochs
        self.stop_training = False
        self.summary.name = summary_name or self.summary.name  # Keep old experiment name if new one not provided
        self.summary.history.pop('test', None)

    def write_summary(self, key: str, value: Any) -> None:
        """Write an entry into the `Summary` object (iff the experiment was named).

        Args:
            key: The key to write into the summary object.
            value: The value to write into the summary object.
        """
        if self.summary:
            self.summary.history[self.mode][key][self.global_step or 0] = value
