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
import gc
import os
import tempfile
from collections import ChainMap
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Set, Tuple, TypeVar, Union

import gdown
import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision_tf
import torch
from tensorflow.python.distribute.values import DistributedValues

from fastestimator.backend.load_model import load_model
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.op.numpyop import NumpyOp, forward_numpyop, RemoveIf
from fastestimator.op.op import get_inputs_by_op, write_outputs_by_op
from fastestimator.op.tensorop.model.update import UpdateOp
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.schedule.schedule import EpochScheduler, RepeatScheduler, Scheduler, get_current_items
from fastestimator.util.traceability_util import trace_model, traceable
from fastestimator.util.util import NonContext, get_batch_size, to_list

Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)
T = TypeVar('T')

GOOGLE_DRIVE_URL = "https://drive.google.com"


@traceable()
class BaseNetwork:
    """A base class for Network objects.

    Networks are used to define the computation graph surrounding one or more models during training.

    Args:
        target_type: What tensor type is expected by this network ('torch' or 'tf').
        ops: The operators to be executed throughout training / testing / inference. These are likely to contain one or
            more model ops, as well as loss ops and update ops.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.

    Raises:
        ValueError: Mixed precision settings for all models are not the same.
    """
    def __init__(
        self,
        target_type: str,
        device: Optional[torch.device],
        ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None
    ) -> None:
        self.ops = to_list(ops)
        self.target_type = target_type
        self.device = device
        for op in get_current_items(self.ops):
            op.build(framework=self.target_type, device=self.device)
        self.models = to_list(_collect_models(ops))
        self.postprocessing = to_list(postprocessing)
        for pop in self.postprocessing:
            if isinstance(pop, RemoveIf):
                raise ValueError("Filtering is currently not supported in network post-processing")
        self._verify_inputs()
        self.effective_inputs = dict()
        self.effective_outputs = dict()
        self.epoch_ops = []
        self.epoch_postprocessing = []
        self.epoch_models = set()
        self.epoch_state = dict()
        self.mixed_precision = any([model.mixed_precision for model in self.models])

        if self.mixed_precision and not all([model.mixed_precision for model in self.models]):
            raise ValueError("Cannot mix full precision and mixed-precision models")

    def _verify_inputs(self) -> None:
        """Ensure that all ops are TensorOps.

        Raises:
            AssertionError: If any of the ops are not TensorOps.
        """
        for op in get_current_items(self.ops):
            assert isinstance(op, TensorOp), "unsupported op format, Network ops must be TensorOps"
        for op in get_current_items(self.postprocessing):
            assert isinstance(op, NumpyOp), "unsupported op format, Network postprocessing must be NumpyOps"

    def get_scheduled_items(self, mode: str) -> List[Any]:
        """Get a list of items considered for scheduling.

        Args:
            mode: Current execution mode.

        Returns:
            List of schedulable items in Network.
        """
        if mode == "train":
            all_items = self.ops + [model.optimizer for model in self.models] + self.postprocessing
        else:
            all_items = self.ops + self.postprocessing
        return all_items

    def load_epoch(self,
                   mode: str,
                   epoch: int,
                   ds_id: Union[str, None],
                   output_keys: Optional[Set[str]] = None,
                   warmup: bool = False,
                   eager: bool = False) -> None:
        """Prepare the network to run a given epoch and mode.

        This method is necessary since schedulers and op mode restrictions may result in different computation graphs
        every epoch.

        Args:
            mode: The mode to prepare to execute. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch to prepare to execute.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.
            output_keys: What keys can be moved from the GPU back to the CPU after executing a step.
            warmup: Whether to prepare to execute it warmup mode or not (end users can likely ignore this argument).
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        self.effective_inputs[mode] = self.get_effective_input_keys(mode, epoch, ds_id)
        self.effective_outputs[mode] = self.get_all_output_keys(mode, epoch, ds_id)
        if output_keys:
            self.effective_outputs[mode] = self.effective_outputs[mode].intersection(
                output_keys) | self._get_effective_postprocessing_input_keys(mode, epoch, ds_id)
        self.epoch_ops = get_current_items(self.ops, mode, epoch, ds_id=ds_id)
        self.epoch_postprocessing = get_current_items(self.postprocessing, mode, epoch, ds_id=ds_id)
        self.epoch_models = set.union(*[op.get_fe_models() for op in self.epoch_ops])
        gradient_ops = [op for op in self.epoch_ops if op.fe_retain_graph() is not None]
        for idx, gradient_op in enumerate(gradient_ops):
            gradient_op.fe_retain_graph(idx != len(gradient_ops) - 1)
        self.epoch_state = {
            "warmup": warmup,
            "mode": mode,
            "req_grad": len(gradient_ops) > 0,
            "epoch": epoch,
            "deferred": {},
            "eager": eager
        }
        # warmup: bool, mode: str, req_grad: bool, epoch: int, deferred: Dict[str, List[Callable]]]
        for model in self.epoch_models:
            if hasattr(model, "optimizer") and model.optimizer is not None:
                if isinstance(model.optimizer, Scheduler):
                    model.current_optimizer = model.optimizer.get_current_value(epoch)
                else:
                    model.current_optimizer = model.optimizer

    def unload_epoch(self) -> None:
        """Clean up the network after running an epoch.
        """
        pass

    def get_loss_keys(self) -> Set[str]:
        """Find all of the keys associated with model losses.

        Returns:
            All of the keys associated with model losses in this network.
        """
        loss_keys = set()
        for op in get_current_items(self.ops):
            loss_keys |= op.get_fe_loss_keys()
        return loss_keys

    def get_effective_input_keys(self, mode: str, epoch: int, ds_id: Optional[str] = None) -> Set[str]:
        """Determine which keys need to be provided as input to the network during the given `epoch`.

        Args:
            mode: The execution mode to consider. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch number to consider for determining inputs.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.

        Returns:
            The necessary inputs for the network to execute the given `epoch` and `mode`.
        """
        input_keys = set()
        produced_keys = set()
        for op in get_current_items(self.ops + self.postprocessing, mode, epoch, ds_id=ds_id):
            input_keys.update(set(key for key in op.inputs if key not in produced_keys))
            produced_keys.update(op.outputs)
        return input_keys

    def _get_effective_postprocessing_input_keys(self, mode: str, epoch: int, ds_id: Optional[str] = None) -> Set[str]:
        """Determine which keys need to be provided as input to the postprocessing during the given `epoch`.

        Args:
            mode: The execution mode to consider. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch number to consider for determining inputs.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.

        Returns:
            The necessary inputs for the postprocessing to execute the given `epoch` and `mode`.
        """
        input_keys = set()
        produced_keys = set()
        for op in get_current_items(self.postprocessing, mode, epoch, ds_id=ds_id):
            input_keys.update(set(key for key in op.inputs if key not in produced_keys))
            produced_keys.update(op.outputs)
        return input_keys

    def get_all_output_keys(self, mode: str, epoch: int, ds_id: Optional[str] = None) -> Set[str]:
        """Get all of the keys that will be generated by the network during the given `epoch` and `mode`.

        Args:
            mode: The execution mode to consider. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch number to consider when searching for outputs.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.

        Returns:
            The keys that will be generated by the network's Ops during the `epoch` for the given `mode`.
        """
        output_keys = set()
        for op in get_current_items(self.ops + self.postprocessing, mode, epoch, ds_id=ds_id):
            output_keys.update(op.outputs)
        return output_keys

    @staticmethod
    def _forward_batch(batch: MutableMapping[str, Any], state: Dict[str, Any], ops: List[TensorOp]) -> None:
        """Run a forward pass through the network's Op chain given a `batch` of data.

        Args:
            batch: A batch of input data. Predictions from the network will be written back into this dictionary.
            state: A dictionary holding information about the current execution context. The TF gradient tape, for
                example will be stored here.
            ops: Which ops to execute.
        """
        for op in ops:
            data = get_inputs_by_op(op, batch)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_op(op, batch, data)
        for fn_list in state['deferred'].values():
            for fn in fn_list:
                fn()
        state['deferred'].clear()

    def run_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # Batch, Prediction
        """Run a forward step through the Network on a batch of data, including postprocessing.

        This method expects that Network.load_epoch() has already been invoked. The return data will be on the CPU.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            (batch_data, prediction_data)
        """
        batch, prediction = self._run_step(batch)
        forward_numpyop(ops=self.epoch_postprocessing,
                        data=ChainMap(prediction, batch),
                        state=self.epoch_state,
                        batched=True)
        return batch, prediction

    def _run_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # Batch, Prediction
        """Run a forward step through the Network on a batch of data, excluding postprocessing.

        Implementations of this method within derived classes should handle bringing the prediction data back from the
        (multi-)GPU environment to the CPU. This method expects that Network.load_epoch() has already been invoked.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            (batch_data, prediction_data)
        """
        raise NotImplementedError

    def transform(self, data: Dict[str, Any], mode: str, epoch: int = 1, ds_id: Optional[
                                                                                      str] = None) -> Dict[str, Any]:
        """Run a forward step through the Network on an element of data.

        Args:
            data: The element to data to use as input.
            mode: The mode in which to run the transform. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch in which to run the transform.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.

        Returns:
            prediction_data overlaid on the input `data`.
        """
        self.load_epoch(mode, epoch, ds_id, warmup=False, eager=True)
        data = to_tensor(data, target_type=self.target_type)
        data, prediction = self.run_step(data)
        self.unload_epoch()
        return {**data, **prediction}


def _collect_models(ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]) -> Set[Model]:
    """Collect all model instances from amongst a list of ops.

    Args:
        ops: The ops to be searched through.

    Returns:
        All of the model instances contained within the `ops`.
    """
    models = set()
    for op in get_current_items(ops):
        models |= op.get_fe_models()
    return models


# noinspection PyPep8Naming
def Network(
        ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]],
        pops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp,
                                                                      Scheduler[NumpyOp]]]] = None) -> BaseNetwork:
    """A function to automatically instantiate the correct Network derived class based on the given `ops`.

    Args:
        ops: A collection of Ops defining the graph for this Network. It should contain at least one ModelOp, and all
            models should be either TensorFlow or Pytorch. We currently do not support mixing TensorFlow and Pytorch
            models within the same network.
        pops: Postprocessing Ops. A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been
            executed. Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than
            single points.

    Returns:
        A network instance containing the given `ops`.

    Raises:
        AssertionError: If TensorFlow and PyTorch models are mixed, or if no models are provided.
        ValueError: If a model is provided whose type cannot be identified as either TensorFlow or PyTorch.
    """
    ops = to_list(ops)
    models = _collect_models(ops)
    framework = set()
    model_names = set()
    for model in models:
        # 'Model' and 'model' should not be considered unique in case you are saving on a non-case-sensitive filesystem
        model_names.add(model.model_name.lower())
        if isinstance(model, tf.keras.Model):
            framework.add("tf")
        elif isinstance(model, torch.nn.Module):
            framework.add("torch")
        else:
            framework.add("unknown")
    if len(framework) == 0:
        framework.add('tf')  # We will use tf as default framework if no models are found
    assert len(framework) == 1, "please make sure either tensorflow or torch model is used in network"
    assert len(model_names) == len(models), "all models must have unique model names"

    framework = framework.pop()
    if framework == "tf":
        network = TFNetwork(ops, pops)
    elif framework == "torch":
        network = TorchNetwork(ops, pops)
    else:
        raise ValueError("Unknown model type")
    return network


@traceable()
class TorchNetwork(BaseNetwork):
    """An extension of BaseNetwork for PyTorch models.

    Args:
        ops: The ops defining the execution graph for this Network.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.

    """
    def __init__(
        self,
        ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None
    ) -> None:
        super().__init__(target_type='torch',
                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                         ops=ops,
                         postprocessing=postprocessing)

    def load_epoch(self,
                   mode: str,
                   epoch: int,
                   ds_id: Union[str, None],
                   output_keys: Optional[Set[str]] = None,
                   warmup: bool = False,
                   eager: bool = False) -> None:
        """Prepare the network to run a given epoch and mode.

        This method is necessary since schedulers and op mode restrictions may result in different computation graphs
        every epoch. This also moves all of the necessary models from the CPU onto the GPU(s).

        Args:
            mode: The mode to prepare to execute. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch to prepare to execute.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.
            output_keys: What keys must be moved from the GPU back to the CPU after executing a step.
            warmup: Whether to prepare to execute it warmup mode or not (end users can likely ignore this argument).
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        super().load_epoch(mode=mode, epoch=epoch, ds_id=ds_id, output_keys=output_keys, warmup=warmup, eager=eager)
        if self.device.type == "cuda":
            for model in self.epoch_models:
                # move model variables to gpu
                model.to(self.device)
                if model.current_optimizer and mode == "train":
                    # move optimizer variables to gpu
                    self._move_optimizer_between_device(model.current_optimizer.state, self.device)
        # Set all of the contiguous final updates to defer their updates by default to enable things like CycleGan
        # This is not necessary for TF because overriding tf weights does not confuse the gradient tape computation
        for op in reversed(self.epoch_ops):
            if isinstance(op, UpdateOp):
                op._old_defer = op.defer
                op.defer = True
            else:
                break

    def _move_optimizer_between_device(self, data: Dict[str, Any], device: Union[str, torch.device]) -> None:
        """Move optimizer state between gpu and cpu recursively.

        Args:
            data: Optimizer state.
            device: The target device.
        """
        for key in data:
            if isinstance(data[key], dict):
                self._move_optimizer_between_device(data[key], device)
            else:
                try:
                    data[key] = data[key].to(device)
                except:
                    pass

    def unload_epoch(self) -> None:
        """Clean up the network after running an epoch.

        In this case we move all of the models from the GPU(s) back to the CPU.
        """
        if self.device.type == "cuda":
            for model in self.epoch_models:
                # move model variables to cpu
                model.to("cpu")
                if model.current_optimizer and self.epoch_state["mode"] == "train":
                    # move optimizer variables to cpu
                    self._move_optimizer_between_device(model.current_optimizer.state, "cpu")
        # Set the final update ops back to their original defer status
        for op in reversed(self.epoch_ops):
            if isinstance(op, UpdateOp):
                op.defer = op.__dict__.get('_old_defer', op.defer)
            else:
                break

    def _get_effective_batch_input(self, batch: MutableMapping[str, Any], mode: str) -> Dict[str, Any]:
        """Copy input data from the the CPU onto the GPU(s).

        This method will filter inputs from the batch so that only data required by the network during execution will be
        copied to the GPU.

        Args:
            batch: The input data to be moved.
            mode: The current execution mode. One of 'train', 'eval', 'test', or 'infer'.

        Returns:
            The input data ready for use on GPU(s).
        """
        if self.device.type == "cuda":
            new_batch = {
                key: self._move_tensor_between_device(batch[key], self.device)
                for key in self.effective_inputs[mode] if key in batch
            }
        else:
            new_batch = {key: batch[key] for key in self.effective_inputs[mode] if key in batch}
        return new_batch

    def _run_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a forward step through the Network on a batch of data.

        Implementations of this method within derived classes should handle bringing the prediction data back from the
        (multi-)GPU environment to the CPU. This method expects that Network.load_epoch() has already been invoked.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            (batch_data, prediction_data)
        """
        mode = self.epoch_state["mode"]
        batch_in = self._get_effective_batch_input(batch, mode)
        self.epoch_state["tape"] = NonContext()
        # gpu operation
        with torch.no_grad() if not self.epoch_state["req_grad"] else NonContext():
            with torch.cuda.amp.autocast() if self.mixed_precision else NonContext():
                self._forward_batch(batch_in, self.epoch_state, self.epoch_ops)

        # copy data to cpu
        if self.device.type == "cuda":
            prediction = {
                key: self._move_tensor_between_device(self._detach_tensor(batch_in[key]), "cpu")
                for key in self.effective_outputs[mode] if key in batch_in
            }
        else:
            prediction = {
                key: self._detach_tensor(batch_in[key])
                for key in self.effective_outputs[mode] if key in batch_in
            }
        return batch, prediction

    def _move_tensor_between_device(self, data: T, device: Union[str, torch.device]) -> T:
        """Move tensor between gpu and cpu recursively.

        Args:
            data: The input data to be moved.
            device: The target device.

        Returns:
            Output data.
        """
        if isinstance(data, dict):
            return {key: self._move_tensor_between_device(value, device) for (key, value) in data.items()}
        elif isinstance(data, list):
            return [self._move_tensor_between_device(val, device) for val in data]
        elif isinstance(data, tuple):
            return tuple([self._move_tensor_between_device(val, device) for val in data])
        elif isinstance(data, set):
            return set([self._move_tensor_between_device(val, device) for val in data])
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def _detach_tensor(self, data: T) -> T:
        """Detach tensor from current graph recursively.

        Args:
            data: The data to be detached.

        Returns:
            Output data.
        """
        if isinstance(data, dict):
            return {key: self._detach_tensor(value) for (key, value) in data.items()}
        elif isinstance(data, list):
            return [self._detach_tensor(val) for val in data]
        elif isinstance(data, tuple):
            return tuple([self._detach_tensor(val) for val in data])
        elif isinstance(data, set):
            return set([self._detach_tensor(val) for val in data])
        elif isinstance(data, torch.Tensor):
            return data.detach()
        return data


@traceable()
class TFNetwork(BaseNetwork):
    """An extension of BaseNetwork for TensorFlow models.

    Args:
        ops: The ops defining the execution graph for this Network.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.
    """
    def __init__(
        self,
        ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None
    ) -> None:
        super().__init__(target_type='tf', device=None, ops=ops, postprocessing=postprocessing)

    def load_epoch(self,
                   mode: str,
                   epoch: int,
                   ds_id: Union[None, str],
                   output_keys: Optional[Set[str]] = None,
                   warmup: bool = False,
                   eager: bool = False) -> None:
        """Prepare the network to run a given epoch and mode.

        This method is necessary since schedulers and op mode restrictions may result in different computation graphs
        every epoch. This also converts the epoch index a tensor to avoid tensorflow graph rebuilding.

        Args:
            mode: The mode to prepare to execute. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch to prepare to execute.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.
            output_keys: What keys must be moved from the GPU back to the CPU after executing a step.
            warmup: Whether to prepare to execute it warmup mode or not (end users can likely ignore this argument).
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        super().load_epoch(mode=mode, epoch=epoch, ds_id=ds_id, output_keys=output_keys, warmup=warmup, eager=eager)
        # Don't cause a re-trace just because epoch changed
        self.epoch_state["epoch"] = tf.convert_to_tensor(self.epoch_state["epoch"])
        # Need to re-trace the TF graph if the optimizer is changing due to scheduling:
        opt_str = "x".join(
            [str(id(model.current_optimizer)) for model in self.epoch_models if hasattr(model, 'current_optimizer')])
        self.epoch_state["_force_tf_retrace"] = hash(opt_str)  # Hash to keep at fixed memory overhead

    def unload_epoch(self) -> None:
        # This prevents a tf graph memory leak that would slow down long trainings. Since we
        # re-build graphs every epoch there is no reason to keep old ones around.
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            return  # TODO - Find a way to clear graph for multi-gpu
        else:
            tf.keras.backend.clear_session()

    def _run_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a forward step through the Network on a batch of data.

        Implementations of this method within derived classes should handle bringing the prediction data back from the
        (multi-)GPU environment to the CPU. This method expects that Network.load_epoch() has already been invoked.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            (batch_data, prediction_data)
        """
        mode = self.epoch_state["mode"]
        batch_in = self._get_effective_batch_input(batch, mode)
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            if self.epoch_state["eager"]:
                prediction = strategy.run(
                    self._forward_step_eager,
                    args=(batch_in, self.epoch_state, self.epoch_ops, to_list(self.effective_outputs[mode])))
            else:
                prediction = strategy.run(
                    self._forward_step_static,
                    args=(batch_in, self.epoch_state, self.epoch_ops, to_list(self.effective_outputs[mode])))
            batch = self._per_replica_to_global(batch)
            prediction = self._per_replica_to_global(prediction)
        else:
            if self.epoch_state["eager"]:
                prediction = self._forward_step_eager(batch_in,
                                                      self.epoch_state,
                                                      self.epoch_ops,
                                                      to_list(self.effective_outputs[mode]))
            else:
                prediction = self._forward_step_static(batch_in,
                                                       self.epoch_state,
                                                       self.epoch_ops,
                                                       to_list(self.effective_outputs[mode]))
        return batch, prediction

    def _per_replica_to_global(self, data: T) -> T:
        """Combine data from "per-replica" values recursively.

        For multi-GPU training, data are distributed using `tf.distribute.Strategy.experimental_distribute_dataset`.
        This method collects data from all replicas and combines them into one.

        Args:
            data: Distributed data.

        Returns:
            Combined data from all replicas.
        """
        if isinstance(data, DistributedValues):
            if data.values[0].shape.rank == 0:
                return tf.reduce_mean(tuple(d for d in data.values if not tf.math.is_nan(d)))
            else:
                return tf.concat(data.values, axis=0)
        elif isinstance(data, dict):
            result = {}
            for key, val in data.items():
                result[key] = self._per_replica_to_global(val)
            return result
        elif isinstance(data, list):
            return [self._per_replica_to_global(val) for val in data]
        elif isinstance(data, tuple):
            return tuple([self._per_replica_to_global(val) for val in data])
        elif isinstance(data, set):
            return set([self._per_replica_to_global(val) for val in data])
        else:
            return data

    def _get_effective_batch_input(self, batch: MutableMapping[str, Any], mode: str) -> Dict[str, Any]:
        """Filter input data so that only the data required by the Network is moved onto the GPU.

        Args:
            batch: An unfiltered batch of input data.
            mode: The current execution mode. One of 'train', 'eval', 'test', or 'infer'.

        Returns:
            The filtered input data ready for use on GPU(s).
        """
        new_batch = {}
        for key in self.effective_inputs[mode]:
            if key in batch:
                new_batch[key] = batch[key]
        return new_batch

    def _forward_step_eager(self,
                            batch: Dict[str, Any],
                            state: Dict[str, Any],
                            ops: List[TensorOp],
                            effective_outputs: List[str]) -> Dict[str, Any]:
        """Run a forward step of the Network in eager (non-static graph) mode.

        Args:
            batch: The input data for the Network.
            state: A dictionary containing information about the current execution environment, including the active
                gradient tape.
            ops: A list of Ops to run during the forward step.
            effective_outputs: Which outputs should be copied from the GPU back onto the CPU for further use in Traces.

        Returns:
            The prediction dictionary resulting from a forward pass of the Network.
        """
        batch = ChainMap({}, batch)
        prediction = {}
        with tf.GradientTape(persistent=True) if state["req_grad"] else NonContext() as tape:
            state['tape'] = tape
            self._forward_batch(batch, state, ops)
        del state['tape']
        del tape
        for key in effective_outputs:
            if key in batch:
                prediction[key] = batch[key]
        return prediction

    @tf.function(experimental_relax_shapes=True)
    def _forward_step_static(self,
                             batch: Dict[str, Any],
                             state: Dict[str, Any],
                             ops: List[TensorOp],
                             effective_outputs: List[str]) -> Dict[str, Any]:
        """Run a forward step of the Network in static graph mode.

        Args:
            batch: The input data for the Network.
            state: A dictionary containing information about the current execution environment, including the active
                gradient tape.
            ops: A list of Ops to run during the forward step.
            effective_outputs: Which outputs should be copied from the GPU back onto the CPU for further use in Traces.

        Returns:
            The prediction dictionary resulting from a forward pass of the Network.
        """
        batch = dict(batch)
        prediction = {}
        with tf.GradientTape(persistent=True) if state["req_grad"] else NonContext() as tape:
            state['tape'] = tape
            self._forward_batch(batch, state, ops)
        del state['tape']
        del tape
        for key in effective_outputs:
            if key in batch:
                prediction[key] = batch[key]
        return prediction

    def transform(self, data: Dict[str, Any], mode: str, epoch: int = 1, ds_id: Optional[str] = None) -> Dict[str, Any]:
        """Run a forward step through the Network on an element of data.

        Args:
            data: The element to data to use as input.
            mode: The mode in which to run the transform. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch in which to run the transform.
            ds_id: The current dataset id. If None, ops with all ds_id will be considered.

        Returns:
            (batch_data, prediction_data)
        """
        # Distribute multi-gpu data for processing
        sub_sample = False
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            batch_size, num_devices = get_batch_size(data), strategy.num_replicas_in_sync
            if batch_size < num_devices:
                data = self._fill_batch(data, num_devices - batch_size)
                sub_sample = True
            data = next(iter(strategy.experimental_distribute_dataset(tf.data.Dataset.from_tensors(data))))
        results = super().transform(data, mode, epoch, ds_id=ds_id)
        if sub_sample:
            results = self._subsample_data(results, batch_size)
        return results

    def _fill_batch(self, data: T, n: int) -> T:
        """Fill data on batch dimension repeating the first n indices at the end.

        Args:
            data: The data to be filled.
            n: The number of times to be repeated.

        Returns:
            Filled data.
        """
        if isinstance(data, dict):
            return {key: self._fill_batch(val, n) for (key, val) in data.items()}
        elif isinstance(data, list):
            return [self._fill_batch(val, n) for val in data]
        elif isinstance(data, tuple):
            return tuple([self._fill_batch(val, n) for val in data])
        elif isinstance(data, set):
            return set([self._fill_batch(val, n) for val in data])
        elif hasattr(data, "shape"):
            paddings = [[0, n]] + [[0, 0] for _ in range(len(data.shape) - 1)]
            return np.pad(data, pad_width=paddings, mode="symmetric")
        else:
            return data

    def _subsample_data(self, data: T, n: int) -> T:
        """Subsample data by selecting the first n indices recursively.

        Args:
            data: The data to be subsampled.
            n: The number of indices to be subsampled.

        Returns:
            Subsampled data.
        """
        if isinstance(data, dict):
            return {key: self._subsample_data(val, n) for (key, val) in data.items()}
        elif isinstance(data, list):
            return [self._subsample_data(val, n) for val in data]
        elif isinstance(data, tuple):
            return tuple([self._subsample_data(val, n) for val in data])
        elif isinstance(data, set):
            return set([self._subsample_data(val, n) for val in data])
        elif hasattr(data, "shape") and list(data.shape) and data.shape[0] > n:
            return data[0:n]
        else:
            return data


def build(model_fn: Callable[[], Union[Model, List[Model]]],
          optimizer_fn: Union[str, Scheduler, Callable, List[str], List[Callable], List[Scheduler], None],
          weights_path: Union[str, None, List[Union[str, None]]] = None,
          model_name: Union[str, List[str], None] = None,
          mixed_precision: bool = False) -> Union[Model, List[Model]]:
    """Build model instances and associate them with optimizers.

    This method can be used with TensorFlow models / optimizers:
    ```python
    model_def = fe.architecture.tensorflow.LeNet
    model = fe.build(model_fn = model_def, optimizer_fn="adam")
    model = fe.build(model_fn = model_def, optimizer_fn=lambda: tf.optimizers.Adam(lr=0.1))
    model = fe.build(model_fn = model_def, optimizer_fn="adam", weights_path="~/weights.h5")
    ```

    This method can be used with PyTorch models / optimizers:
    ```python
    model_def = fe.architecture.pytorch.LeNet
    model = fe.build(model_fn = model_def, optimizer_fn="adam")
    model = fe.build(model_fn = model_def, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=0.1))
    model = fe.build(model_fn = model_def, optimizer_fn="adam", weights_path="~/weights.pt)
    ```

    Args:
        model_fn: A function that define model(s).
        optimizer_fn: Optimizer string/definition or a list of optimizer instances/strings. The number of optimizers
            provided here should match the number of models generated by the `model_fn`.
        model_name: Name(s) of the model(s) that will be used for logging purpose. If None, a name will be
            automatically generated and assigned.
        weights_path: Path(s) from which to load model weights. If not None, then the number of weight paths provided
            should match the number of models generated by the `model_fn`.
        mixed_precision: Whether to enable mixed-precision network operations.

    Returns:
        models: The model(s) built by FastEstimator.
    """
    def _generate_model_names(num_names):
        names = ["model" if i + build.count == 0 else "model{}".format(i + build.count) for i in range(num_names)]
        build.count += num_names
        return names

    if not hasattr(build, "count"):
        build.count = 0
    # The following garbage collection is needed for if a TF model was running, but then died due to an exception being
    # thrown, but the exception was then caught, whereupon the user wanted to switch to a pytorch model instead. Absent
    # this collection, you would see: "Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization error". This
    # would be followed by the death of the pytorch multi-processor which would report something like the following:
    # RuntimeError: DataLoader worker (pid 4225) is killed by signal: Aborted.
    # RuntimeError: DataLoader worker (pid(s) 4225, 4226, 4227) exited unexpectedly
    gc.collect()
    # tensorflow models requires setting global policies prior to model creation. Since there is no way to know the
    # framework of model, setting the policy for both tf and pytorch here.
    if mixed_precision:
        mixed_precision_tf.set_global_policy(mixed_precision_tf.Policy('mixed_float16'))
    else:
        mixed_precision_tf.set_global_policy(mixed_precision_tf.Policy('float32'))
    if torch.cuda.device_count() > 1:
        if not isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy):
            tf.distribute.experimental_set_strategy(tf.distribute.MirroredStrategy())
    models, optimizer_fn = to_list(model_fn()), to_list(optimizer_fn)
    # fill optimizers if optimizer_fn is None
    if not optimizer_fn:
        optimizer_fn = [None] * len(models)
    # generate names
    if not model_name:
        model_name = _generate_model_names(len(models))
    model_name = to_list(model_name)
    # load weights
    if weights_path:
        weights_path = to_list(weights_path)
    else:
        weights_path = [None] * len(models)
    assert len(models) == len(optimizer_fn) == len(weights_path) == len(model_name), \
        "Found inconsistency in number of models, optimizers, model_name or weights"
    # create optimizer
    for idx, (model, optimizer_def, weight, name) in enumerate(zip(models, optimizer_fn, weights_path, model_name)):
        models[idx] = trace_model(_fe_compile(model, optimizer_def, weight, name, mixed_precision),
                                  model_idx=idx if len(models) > 1 else -1,
                                  model_fn=model_fn,
                                  optimizer_fn=optimizer_def,
                                  weights_path=weight)
    if len(models) == 1:
        models = models[0]
    return models


def _fe_compile(model: Model,
                optimizer_fn: Union[str, Scheduler, Callable, None],
                weight: Union[str, None],
                name: str,
                mixed_precision: bool) -> Model:
    """A function to bundle models with their optimizers.

    Args:
        model: The model to be bundled.
        optimizer_fn: The optimizer to be associated with the given `model`.
        weight: A path to weights to be associated with the `model`.
        name: The name of the model.
        mixed_precision: Whether to enable mixed-precision training.

    Returns:
        The `model` combined with its optimizer, weights, and name. Models will also have an 'fe_compiled' flag to
        indicate that they were built via this function.
    """
    if isinstance(model, tf.keras.Model):
        framework = "tf"
    elif isinstance(model, torch.nn.Module):
        framework = "torch"
    else:
        raise ValueError("unrecognized model format: {}".format(type(model)))
    # torch multi-gpu handling
    if framework == "torch" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # mark models with its mixed_precision flag
    model.mixed_precision = mixed_precision
    if isinstance(optimizer_fn, EpochScheduler):
        for epoch, optimizer_def in optimizer_fn.epoch_dict.items():
            optimizer_fn.epoch_dict[epoch] = _build_optimizer(optimizer_def, model, framework, mixed_precision)
        model.current_optimizer = optimizer_fn.get_current_value(1)
    elif isinstance(optimizer_fn, RepeatScheduler):
        for idx, optimizer_def in enumerate(optimizer_fn.repeat_list):
            optimizer_fn.repeat_list[idx] = _build_optimizer(optimizer_def, model, framework, mixed_precision)
        model.current_optimizer = optimizer_fn.get_current_value(1)
    else:
        optimizer_fn = _build_optimizer(optimizer_fn, model, framework, mixed_precision)
        model.current_optimizer = optimizer_fn
    if framework == "tf":
        model.input_spec = None  # this is to handle a behavior change in tf 2.4.1 that enforces a input shape check
    model.optimizer = optimizer_fn
    model.fe_compiled = True
    if weight:
        if weight.startswith(GOOGLE_DRIVE_URL):
            tmp_dir = tempfile.mkdtemp()
            file_name = gdown.download(weight, quiet=False)
            os.rename(os.path.join('./', file_name), os.path.join(tmp_dir, file_name))
            weight = gdown.download(weight, os.path.join(tmp_dir, file_name), quiet=False)
        load_model(model, weight)
    model.model_name = name
    return model


def _build_optimizer(optimizer_fn: Union[str, Callable, None], model: Model, framework: str,
                     mixed_precision: bool) -> Union[None, tf.optimizers.Optimizer, torch.optim.Optimizer]:
    """A helper method to instantiate an optimizer.

    Args:
        optimizer_fn: The function responsible for constructing an optimizer, or else a string indicating one of the
            default optimizers to be used.
        model: The model to associate the optimizer with.
        framework: Which backend framework should be used ('tf' or 'torch').
        mixed_precision: Whether to enable mixed-precision training.

    Returns:
        An optimizer instance, or None if `optimizer_fn` was None.
    """
    if isinstance(optimizer_fn, str):
        optimizer_fn = _optimizer_fn_from_string(optimizer_fn, framework)
    optimizer = _optimizer_fn_to_optimizer(optimizer_fn, model, framework, mixed_precision)
    return optimizer


def _optimizer_fn_from_string(name: str, framework: str) -> Callable:
    """A function to construct default optimizers based on string keys.

    Args:
        name: The name of the default optimizer to instantiate.
        framework: Which backend framework should be used ('tf' or 'torch').

    Returns:
        An optimizer instance corresponding to the given `name` and `framework`.
    """
    tf_optimizer_fn = {
        'adadelta': tf.optimizers.Adadelta,
        'adagrad': tf.optimizers.Adagrad,
        'adam': tf.optimizers.Adam,
        'adamax': tf.optimizers.Adamax,
        'rmsprop': tf.optimizers.RMSprop,
        'sgd': tf.optimizers.SGD
    }
    pytorch_optimizer_fn = {
        'adadelta': lambda x: torch.optim.Adadelta(params=x),
        'adagrad': lambda x: torch.optim.Adagrad(params=x),
        'adam': lambda x: torch.optim.Adam(params=x),
        'adamax': lambda x: torch.optim.Adamax(params=x),
        'rmsprop': lambda x: torch.optim.RMSprop(params=x),
        'sgd': lambda x: torch.optim.SGD(params=x, lr=0.01)
    }
    if framework == "tf":
        optimizer_fn = tf_optimizer_fn[name]
    else:
        optimizer_fn = pytorch_optimizer_fn[name]
    return optimizer_fn


def _optimizer_fn_to_optimizer(optimizer_fn: Union[Callable, None], model: Model, framework: str,
                               mixed_precision: bool) -> Union[None, tf.optimizers.Optimizer, torch.optim.Optimizer]:
    """A helper function to invoke an optimizer function.

    Args:
        optimizer_fn: The function to be invoked in order to instantiate an optimizer.
        model: The model with which the optimizer should be associated.
        framework: Which backend framework should be used ('tf' or 'torch').
        mixed_precision: Whether to enable mixed-precision training.

    Returns:
        An optimizer instance, or None if `optimizer_fn` was None.
    """
    optimizer = None
    if optimizer_fn:
        if framework == "tf":
            try:
                optimizer = optimizer_fn()
            except:
                raise AssertionError("optimizer_fn of Tensorflow backend should be callable without args. Please "
                                     "make sure model and optimizer_fn are using the same backend")
            # initialize optimizer variables
            _ = optimizer.iterations
            optimizer._create_hypers()
            optimizer._create_slots(model.trainable_variables)
            # handle mixed precision loss scaling
            if mixed_precision:
                optimizer = mixed_precision_tf.LossScaleOptimizer(optimizer)
            assert isinstance(optimizer, tf.optimizers.Optimizer), "optimizer_fn should generate tensorflow optimizer"
        else:
            try:
                optimizer = optimizer_fn(model.parameters())
            except Exception as e:
                print("optimizer_fn of Pytorch backend should be callable with single arg. Please sure model and \
                optimizer_fn are using the same backend")
                raise ValueError(repr(e))
            assert isinstance(optimizer, torch.optim.Optimizer), "optimizer_fn should generate pytorch optimizer"
            if mixed_precision:
                setattr(optimizer, "scaler", torch.cuda.amp.GradScaler())
            else:
                setattr(optimizer, "scaler", None)

    return optimizer
