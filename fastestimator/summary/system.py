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
import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union

import dill as pickle  # Need to use dill since tf.Variable is a weakref object on multi-gpu machines
import tensorflow as tf
import torch

from fastestimator.backend.load_model import load_model
from fastestimator.backend.save_model import save_model
from fastestimator.network import BaseNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler
from fastestimator.summary.summary import Summary
from fastestimator.util.traceability_util import FeSummaryTable, is_restorable

if TYPE_CHECKING:
    from fastestimator.trace.trace import Trace

Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


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
        ds_id: The current dataset id, None if there is only one dataset in each mode.
        exp_id: A unique identifier for current training experiment.
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
        custom_graphs: A place to store extra graphs which are too complicated for the primary history.
    """

    mode: Optional[str]
    ds_id: Optional[str]
    exp_id: int
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
    custom_graphs: Dict[str, List[Summary]]

    def __init__(self,
                 network: BaseNetwork,
                 pipeline: Pipeline,
                 traces: List[Union['Trace', Scheduler['Trace']]],
                 mode: Optional[str] = None,
                 ds_id: Optional[str] = None,
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
        self.ds_id = ds_id
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.batch_idx = None
        self.max_train_steps_per_epoch = max_train_steps_per_epoch
        self.max_eval_steps_per_epoch = max_eval_steps_per_epoch
        self.stop_training = False
        self.summary = Summary(None, system_config)
        self.experiment_time = ""
        self.custom_graphs = {}
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize the training state.
        """
        self.global_step = None
        self.epoch_idx = 0
        # Get a 64 bit random id related to current time
        self.exp_id = int.from_bytes(uuid.uuid1().bytes, byteorder='big', signed=True) >> 64

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
        self.ds_id = None
        self._initialize_state()
        self.batch_idx = None
        self.stop_training = False
        self.summary = Summary(summary_name, system_config)
        self.custom_graphs = {}

    def reset_for_test(self, summary_name: Optional[str] = None) -> None:
        """Partially reset the current `System` object for a new round of testing.

        Args:
            summary_name: The name of the experiment. If not provided, the system will re-use the previous summary name.
        """
        self.experiment_time = self.experiment_time or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = "test"
        self.ds_id = None
        if not self.stop_training:
            self.epoch_idx = self.total_epochs
        self.stop_training = False
        self.summary.name = summary_name or self.summary.name  # Keep old experiment name if new one not provided
        self.summary.history.pop('test', None)
        for graph_set in self.custom_graphs.values():
            for graph in graph_set:
                graph.history.pop('test', None)

    def write_summary(self, key: str, value: Any) -> None:
        """Write an entry into the `Summary` object (iff the experiment was named).

        Args:
            key: The key to write into the summary object.
            value: The value to write into the summary object.
        """
        if self.summary:
            self.summary.history[self.mode][key][self.global_step or 0] = value

    def add_graph(self, graph_name: str, graph: Union[Summary, List[Summary]]) -> None:
        """Write custom summary graphs into the System.

        This can be useful for things like the LabelTracker trace to interact with Traceability reports.

        Args:
            graph_name: The name of the graph (so that you can override it later if desired).
            graph: The custom summary to be tracked.
        """
        if isinstance(graph, Summary):
            self.custom_graphs[graph_name] = [graph]
        else:
            self.custom_graphs[graph_name] = list(graph)

    def save_state(self, save_dir: str) -> None:
        """Load training state.

        Args:
            save_dir: The directory into which to save the state
        """
        os.makedirs(save_dir, exist_ok=True)
        # Start with the high-level info. We could use pickle for this but having it human readable is nice.
        state = {key: value for key, value in self.__dict__.items() if is_restorable(value)[0]}
        with open(os.path.join(save_dir, 'system.json'), 'w') as fp:
            json.dump(state, fp, indent=4)
        # Save all of the models / optimizer states
        for model in self.network.models:
            save_model(model, save_dir=save_dir, save_optimizer=True)
        # Save everything else
        objects = {
            'summary': self.summary,
            'custom_graphs': self.custom_graphs,
            'traces': [trace.__getstate__() if hasattr(trace, '__getstate__') else {} for trace in self.traces],
            'tops': [op.__getstate__() if hasattr(op, '__getstate__') else {} for op in self.network.ops],
            'pops': [op.__getstate__() if hasattr(op, '__getstate__') else {} for op in self.network.postprocessing],
            'nops': [op.__getstate__() if hasattr(op, '__getstate__') else {} for op in self.pipeline.ops],
            'ds': {
                mode: {key: value.__getstate__()
                       for key, value in ds.items() if hasattr(value, '__getstate__')}
                for mode,
                ds in self.pipeline.data.items()
            }
        }
        with open(os.path.join(save_dir, 'objects.pkl'), 'wb') as file:
            pickle.dump(objects, file)

    def load_state(self, load_dir: str) -> None:
        """Load training state.

        Args:
            load_dir: The directory from which to reload the state.

        Raises:
            FileNotFoundError: If necessary files can not be found.
        """
        # Reload the high-level system information
        system_path = os.path.join(load_dir, 'system.json')
        if not os.path.exists(system_path):
            raise FileNotFoundError(f"Could not find system summary file at {system_path}")
        with open(system_path, 'r') as fp:
            state = json.load(fp)
        self.__dict__.update(state)
        # Reload the models
        for model in self.network.models:
            self._load_model(model, load_dir)
        # Reload everything else
        objects_path = os.path.join(load_dir, 'objects.pkl')
        if not os.path.exists(objects_path):
            raise FileNotFoundError(f"Could not find the objects summary file at {objects_path}")
        with open(objects_path, 'rb') as file:
            objects = pickle.load(file)
        self.summary.__dict__.update(objects['summary'].__dict__)
        self.custom_graphs = objects['custom_graphs']
        self._load_list(objects, 'traces', self.traces)
        self._load_list(objects, 'tops', self.network.ops)
        self._load_list(objects, 'pops', self.network.postprocessing)
        self._load_list(objects, 'nops', self.pipeline.ops)
        self._load_dict(objects, 'ds', self.pipeline.data)

    @staticmethod
    def _load_model(model: Model, base_path: str) -> None:
        """Load model and optimizer weights from disk.

        Args:
            model: The model to be loaded.
            base_path: The folder where the model should be located.

        Raises:
            ValueError: If the model is of an unknown type.
            FileNotFoundError: If the model weights or optimizer state is missing.
        """
        if isinstance(model, tf.keras.Model):
            model_ext, optimizer_ext = 'h5', 'pkl'
        elif isinstance(model, torch.nn.Module):
            model_ext, optimizer_ext = 'pt', 'pt'
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        weights_path = os.path.join(base_path, f"{model.model_name}.{model_ext}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Cannot find model weights file at {weights_path}")
        optimizer_path = os.path.join(base_path, f"{model.model_name}_opt.{optimizer_ext}")
        if not os.path.exists(optimizer_path):
            raise FileNotFoundError(f"Cannot find model optimizer file at {optimizer_path}")
        load_model(model, weights_path=weights_path, load_optimizer=True)

    @staticmethod
    def _load_list(states: Dict[str, Any], state_key: str, in_memory_objects: List[Any]) -> None:
        """Load a list of pickled states from the disk.

        Args:
            states: The states to be restored.
            state_key: Which state to select from the dictionary.
            in_memory_objects: The existing in memory objects to be updated.

        Raises:
            ValueError: If the number of saved states does not match the number of in-memory objects.
        """
        states = states[state_key]
        if not isinstance(states, list):
            raise ValueError(f"Expected {state_key} to contain a list, but found a {type(states)}")
        if len(states) != len(in_memory_objects):
            raise ValueError("Expected saved {} to contain {} objects, but found {} instead".format(
                state_key, len(in_memory_objects), len(states)))
        for obj, state in zip(in_memory_objects, states):
            if hasattr(obj, '__setstate__'):
                obj.__setstate__(state)
            elif hasattr(obj, '__dict__'):
                obj.__dict__.update(state)
            else:
                # Might be a None or something else that can't be updated
                pass

    @staticmethod
    def _load_dict(states: Dict[str, Any], state_key: str, in_memory_objects: Dict[Any, Any]) -> None:
        """Load a dictionary of pickled states from the disk.

        Args:
            states: The states to be restored.
            state_key: Which state to select from the dictionary.
            in_memory_objects: The existing in memory objects to be updated.

        Raises:
            ValueError: If the configuration of saved states does not match the number of in-memory objects.
            FileNotFoundError: If the desired state file cannot be found.
        """
        states = states[state_key]
        if not isinstance(states, dict):
            raise ValueError(f"Expected {state_key} to contain a dict, but found a {type(states)}")
        # Note that not being a subset is different from being a superset
        if not states.keys() <= in_memory_objects.keys():
            raise ValueError("Saved {} contained unexpected keys: {}".format(state_key,
                                                                             states.keys() - in_memory_objects.keys()))
        for key, state in states.items():
            obj = in_memory_objects[key]
            if hasattr(obj, '__setstate__'):
                obj.__setstate__(state)
            elif hasattr(obj, '__dict__'):
                obj.__dict__.update(state)
            elif isinstance(obj, dict):
                [System._load_dict(states, k, obj) for k in states]
            else:
                # Might be a None or something else that can't be updated
                pass
