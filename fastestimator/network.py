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
import sys
import tempfile
from collections import ChainMap
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple, Type, TypeVar, \
    Union, overload

import gdown
import keras.mixed_precision as mixed_precision_tf
import numpy as np
import tensorflow as tf
import torch
from keras.engine.sequential import Sequential
from keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer, LossScaleOptimizerV3
from tensorflow.python.distribute.values import DistributedValues
from typing_extensions import Self

import fastestimator as fe
from fastestimator.backend._load_model import load_model
from fastestimator.backend._to_tensor import to_tensor
from fastestimator.op.numpyop import Batch
from fastestimator.op.numpyop import Delete as DeleteNP
from fastestimator.op.numpyop import NumpyOp, RemoveIf, forward_numpyop
from fastestimator.op.op import get_inputs_by_op, write_outputs_by_op
from fastestimator.op.tensorop.model.update import UpdateOp
from fastestimator.op.tensorop.tensorop import Delete, TensorOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import EpochScheduler, RepeatScheduler, Scheduler, get_current_items
from fastestimator.slicer.slicer import Slicer, forward_slicers, reverse_slicers, sanity_assert_slicers
from fastestimator.types import Array, Model
from fastestimator.util.base_util import NonContext, filter_nones, to_list, warn
from fastestimator.util.traceability_util import trace_model, traceable
from fastestimator.util.util import Suppressor, detach_tensors, get_batch_size, get_device, get_num_gpus, \
    move_tensors_to_device

T = TypeVar('T')

GOOGLE_DRIVE_URL = "https://drive.google.com"
_MAC_BUILD_WARNING = False


@traceable(blacklist=('ctx_lock', ))
class BaseNetwork:
    """A base class for Network objects.

    Networks are used to define the computation graph surrounding one or more models during training.

    Args:
        target_type: What tensor type is expected by this network ('torch' or 'tf').
        ops: The operators to be executed throughout training / testing / inference. These are likely to contain one or
            more model ops, as well as loss ops and update ops.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.
        slicers: Slicers to use if you want to cut apart a single batch of data into multiple slices in order to fit
            them onto a smaller GPU. After cutting the data apart and running through the `ops`, the samples are fused
            back together into a single batch on the CPU before being handed over to the `pops`.

    Raises:
        ValueError: Mixed precision settings for all models are not the same.
    """
    def __init__(
        self,
        target_type: str,
        device: Optional[torch.device],
        ops: Sequence[Union[None, TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Sequence[Union[None, NumpyOp,
                                                                                Scheduler[NumpyOp]]]] = None,
        slicers: Union[None, Slicer, Scheduler[Slicer], Sequence[Union[None, Slicer, Scheduler[Slicer]]]] = None,
    ) -> None:
        self.ops = filter_nones(to_list(ops))
        self.target_type = target_type
        self.device = device
        for op in get_current_items(self.ops):
            op.build(framework=self.target_type, device=self.device)
        self.models = to_list(_collect_models(self.ops))
        self.mixed_precision = any([model.mixed_precision for model in self.models])
        if self.mixed_precision and not all([model.mixed_precision for model in self.models]):
            raise ValueError("Cannot mix full precision and mixed-precision models")
        self.postprocessing = filter_nones(to_list(postprocessing))
        for pop in self.postprocessing:
            if isinstance(pop, RemoveIf):
                raise ValueError("Filtering is currently not supported in network post-processing")
            if isinstance(pop, Batch):
                raise ValueError("Post-processing data is already batched, so Batch Op is not supported here.")
        self.slicers = filter_nones(to_list(slicers))
        self._verify_inputs()
        # Per-Epoch/Mode/DS-ID Variables
        self.ctx_lock = Lock()
        self.ctx_inputs: Set[str] = set()
        self.ctx_gpu_inputs: Set[str] = set()  # The inputs needed by TensorOps specifically
        self.ctx_outputs: Set[str] = set()
        self.ctx_ops: List[TensorOp] = []
        self.ctx_postprocessing: List[NumpyOp] = []
        self.ctx_slicers: List[Slicer] = []
        self.ctx_models: Set[Model] = set()
        self.ctx_state: Dict[str, Any] = dict()

    def _verify_inputs(self) -> None:
        """Ensure that all ops are TensorOps.

        Raises:
            AssertionError: If any of the ops are not TensorOps.
        """
        for op in get_current_items(self.ops):
            assert isinstance(op, TensorOp), "unsupported op format, Network ops must be TensorOps"
        for op in get_current_items(self.postprocessing):
            assert isinstance(op, NumpyOp), "unsupported op format, Network postprocessing must be NumpyOps"
        for slicer in get_current_items(self.slicers):
            assert isinstance(slicer, Slicer), f"unsupported slicer object detected of type: {type(slicer)}"

    def get_scheduled_items(self, mode: str) -> List[Any]:
        """Get a list of items considered for scheduling.

        Args:
            mode: Current execution mode.

        Returns:
            List of schedulable items in Network.
        """
        if mode == "train":
            all_items = self.ops + [model.optimizer for model in self.models] + self.postprocessing + self.slicers
        else:
            all_items = self.ops + self.postprocessing + self.slicers
        return all_items

    def __call__(self,
                 mode: str,
                 epoch: int,
                 ds_id: str,
                 desired_output_keys: Optional[Set[str]] = None,
                 warmup: bool = False,
                 eager: bool = False) -> Self:
        # Make sure that a network isn't currently instantiated with other settings
        acquired = self.ctx_lock.acquire(blocking=False)
        if not acquired:
            raise ValueError("You cannot invoke a Network's __call__ method while it is already active.")
        self.ctx_inputs, self.ctx_gpu_inputs, self.ctx_outputs = self._get_ctx_inputs_and_outputs(
            mode, epoch, ds_id, desired_keys=desired_output_keys)
        self.ctx_ops = get_current_items(self.ops, mode, epoch, ds_id=ds_id)
        self.ctx_postprocessing = get_current_items(self.postprocessing, mode, epoch, ds_id=ds_id)
        self.ctx_slicers = get_current_items(self.slicers, mode, epoch, ds_id=ds_id)
        sanity_assert_slicers(self.ctx_slicers)
        self.ctx_models = set.union(*[op.get_fe_models() for op in self.ctx_ops])
        gradient_ops = [op for op in self.ctx_ops if op.fe_retain_graph() is not None]
        for idx, gradient_op in enumerate(gradient_ops):
            gradient_op.fe_retain_graph(idx != len(gradient_ops) - 1)
        self.ctx_state = {
            "warmup": warmup,
            "mode": mode,
            "req_grad": len(gradient_ops) > 0,
            "epoch": epoch,
            "deferred": {},
            "eager": eager
        }
        # warmup: bool, mode: str, req_grad: bool, epoch: int, deferred: Dict[str, List[Callable]]]
        for model in self.ctx_models:
            if hasattr(model, "optimizer") and model.optimizer is not None:
                if isinstance(model.optimizer, Scheduler):
                    model.current_optimizer = model.optimizer.get_current_value(epoch)
                else:
                    model.current_optimizer = model.optimizer
        self.ctx_lock.release()
        return self

    def __enter__(self) -> Self:
        acquired = self.ctx_lock.acquire(blocking=False)
        if not acquired:
            raise ValueError("This network is already in use.")
        return self

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        """Clean up the network after running an epoch.
        """
        self.ctx_inputs = set()
        self.ctx_outputs = set()
        self.ctx_gpu_inputs = set()
        self.ctx_ops = []
        self.ctx_postprocessing = []
        self.ctx_slicers = []
        self.ctx_models = set()
        self.ctx_state = dict()

        self.ctx_lock.release()

    def get_loss_keys(self) -> Set[str]:
        """Find all of the keys associated with model losses.

        Returns:
            All of the keys associated with model losses in this network.
        """
        loss_keys = set()
        for op in get_current_items(self.ops):
            loss_keys |= op.get_fe_loss_keys()
        return loss_keys

    def _get_ctx_inputs_and_outputs(self,
                                    mode: str,
                                    epoch: int,
                                    ds_id: str = '',
                                    desired_keys: Optional[Set[str]] = None) -> Tuple[Set[str], Set[str], Set[str]]:
        """Figure out the Network's input and output keys for the current mode/epoch/ds_id.

        Args:
            mode: The execution mode to consider. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch number to consider for determining inputs.
            ds_id: The current dataset id.
            desired_keys: Which outputs do you actually want returned from the network for further processing.

        Returns:
            A tuple consisting of:
                1) The necessary inputs for the network to execute
                2) The inputs which need to be given to the GPU ops
                3) The outputs the network will return
        """
        network_input_keys = set()
        gpu_input_keys = set()
        produced_keys = set()
        slice_inputs = set()
        unslice_inputs = set()
        pops_inputs = set()
        pops_produced_keys = set()
        for slicer in get_current_items(self.slicers, mode, epoch, ds_id=ds_id):
            network_input_keys.update(set(slicer.slice_inputs))
            unslice_inputs.update(set(slicer.unslice_inputs))
        slice_inputs.update(network_input_keys)
        for op in get_current_items(self.ops, mode, epoch, ds_id=ds_id):
            unsatisfied_inputs = set(key for key in op.inputs if key not in produced_keys)
            network_input_keys.update(unsatisfied_inputs)
            gpu_input_keys.update(unsatisfied_inputs)
            produced_keys.update(op.outputs)
            if isinstance(op, Delete):
                for inp in op.inputs:
                    produced_keys.discard(inp)
        network_input_keys.update(unslice_inputs - produced_keys)
        for op in get_current_items(self.postprocessing, mode, epoch, ds_id=ds_id):
            network_input_keys.update(set(key for key in op.inputs if key not in produced_keys))
            produced_keys.update(op.outputs)
            pops_inputs.update(set(key for key in op.inputs if key not in pops_produced_keys))
            pops_produced_keys.update(op.outputs)
            if isinstance(op, DeleteNP):
                for inp in op.inputs:
                    produced_keys.discard(inp)
                    pops_produced_keys.discard(inp)
        # Figure out outputs
        output_keys = produced_keys
        if desired_keys:
            # If pops require a key then we can't throw it away on the GPU, even if Traces won't use that key later
            output_keys = (output_keys & desired_keys) | pops_inputs
        if slice_inputs:
            # You want the key (output_keys) AND you sliced the key (slice_inputs | produced_keys) but you didn't
            # unslice it
            sliced_but_not_unsliced = (output_keys & (slice_inputs | produced_keys)) - unslice_inputs
            if sliced_but_not_unsliced:
                state = {'epoch': epoch, 'mode': mode, 'ds_id': ds_id}
                raise ValueError(
                    "Since you are using Slicers, you must specify how you would like the following keys to be" +
                    f" un-sliced during {state}: {sliced_but_not_unsliced}")

        return network_input_keys, gpu_input_keys, output_keys

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
            if isinstance(op, Delete):
                for key in op.inputs:
                    del batch[key]
            if op.outputs:
                write_outputs_by_op(op, batch, data)
        for fn_list in state['deferred'].values():
            for fn in fn_list:
                fn()
        state['deferred'].clear()

    def run_step(self, batch: Dict[str, Array]) -> Dict[str, Array]:  # Batch, Prediction
        """Run a forward step through the Network on a batch of data, including postprocessing.

        Usage:

        ```python
        with network(epoch=1, mode='train', ds_id=''):
            network.run_step()
        ```

        The return data will be on the CPU.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            The input data along with prediction data (input keys may be overwritten/obscured).
        """
        if not self.ctx_lock.locked:
            raise ValueError("To invoke the run_step method you must first enter the network (see the doc string)")
        if not self.ctx_state:
            raise ValueError("To invoke the run_step method you must first configure the network (see the doc string)")
        if self.ctx_slicers:
            minibatches = forward_slicers(self.ctx_slicers, data=batch)
            results: List[Tuple[Dict[str, Array], Dict[str, Array]]] = []
            for minibatch in minibatches:
                results.append(self._run_step(minibatch))
            batch = reverse_slicers(self.ctx_slicers,
                                    data=[ChainMap(result[1], result[0]) for result in results],
                                    original_data=batch)
        else:
            batch, prediction = self._run_step(batch)
            batch = ChainMap(prediction, batch)
        forward_numpyop(ops=self.ctx_postprocessing, data=batch, state=self.ctx_state, batched=self.target_type)
        return batch

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

    @overload
    def transform(self, data: Dict[str, Array], mode: str, epoch: int = 1, ds_id: str = '') -> Dict[str, Array]:
        ...

    @overload
    def transform(self, data: Iterable[Dict[str, Array]], mode: str, epoch: int = 1,
                  ds_id: str = '') -> List[Dict[str, Array]]:
        ...

    @overload
    def transform(self, data: Pipeline, mode: str, epoch: int = 1, ds_id: str = '') -> List[Dict[str, Array]]:
        ...

    def transform(self,
                  data: Union[Dict[str, Array], Iterable[Dict[str, Array]], Pipeline],
                  mode: str,
                  epoch: int = 1,
                  ds_id: str = '') -> Union[Dict[str, Array], List[Dict[str, Array]]]:
        """Run a forward step through the Network on one or more elements of data.

        Args:
            data: The data to use as input (or a pipeline to process an entire dataset at once).
            mode: The mode in which to run the transform. One of 'train', 'eval', 'test', or 'infer'.
            epoch: The epoch in which to run the transform.
            ds_id: The current dataset id.

        Returns:
            prediction_data overlaid on the input `data`.
        """
        results = []
        with self(mode=mode, epoch=epoch, ds_id=ds_id, warmup=False, eager=isinstance(data, dict) and not self.slicers):
            if isinstance(data, Pipeline):
                with data(mode=mode, epoch=epoch, ds_id=ds_id, shuffle=False) as loader:
                    for batch in loader:
                        batch = to_tensor(batch, target_type=self.target_type)
                        results.append(self._do_transform(batch))
            else:
                batches = to_list(data)
                for batch in batches:
                    batch = to_tensor(batch, target_type=self.target_type)
                    results.append(self._do_transform(batch))
        if isinstance(data, dict):
            return results[0]
        return results

    def _do_transform(self, batch: Dict[str, Array]) -> Dict[str, Array]:
        """A handle to allow subclasses to modify the behavior of the transform method before it calls run_step

        Args:
            batch: A single batch of data on which to run.

        Returns:
            The predictions overlaid on the input batch dictionary.
        """
        return self.run_step(batch)


def _collect_models(
    ops: Union[None, TensorOp, Scheduler[TensorOp], Iterable[Union[None, TensorOp,
                                                                   Scheduler[TensorOp]]]]) -> Set[Model]:
    """Collect all model instances from amongst a list of ops.

    Args:
        ops: The ops to be searched through.

    Returns:
        All of the model instances contained within the `ops`.
    """
    models = set()
    ops_list = filter_nones(to_list(ops))
    for op in get_current_items(ops_list):
        models |= op.get_fe_models()
    return models


# noinspection PyPep8Naming
def Network(
    ops: Sequence[Union[None, TensorOp, Scheduler[TensorOp]]],
    pops: Union[None, NumpyOp, Scheduler[NumpyOp], Sequence[Union[None, NumpyOp, Scheduler[NumpyOp]]]] = None,
    slicers: Union[None, Slicer, Scheduler[Slicer], Sequence[Union[None, Slicer, Scheduler[Slicer]]]] = None
) -> BaseNetwork:
    """A function to automatically instantiate the correct Network derived class based on the given `ops`.

    Args:
        ops: A collection of Ops defining the graph for this Network. It should contain at least one ModelOp, and all
            models should be either TensorFlow or Pytorch. We currently do not support mixing TensorFlow and Pytorch
            models within the same network.
        pops: Postprocessing Ops. A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been
            executed. Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than
            single points.
        slicers: Slicers to use if you want to cut apart a single batch of data into multiple slices in order to fit
            them onto a smaller GPU. After cutting the data apart and running through the `ops`, the samples are fused
            back together into a single batch on the CPU before being handed over to the `pops`.

    Returns:
        A network instance containing the given `ops`.

    Raises:
        AssertionError: If TensorFlow and PyTorch models are mixed, or if no models are provided.
        ValueError: If a model is provided whose type cannot be identified as either TensorFlow or PyTorch.
    """
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
        network = TFNetwork(ops, pops, slicers)
    elif framework == "torch":
        network = TorchNetwork(ops, pops, slicers)
    else:
        raise ValueError("Unknown model type")
    return network


@traceable(blacklist=('ctx_lock', ))
class TorchNetwork(BaseNetwork):
    """An extension of BaseNetwork for PyTorch models.

    Args:
        ops: The ops defining the execution graph for this Network.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.
        slicers: Slicers to use if you want to cut apart a single batch of data into multiple slices in order to fit
            them onto a smaller GPU. After cutting the data apart and running through the `ops`, the samples are fused
            back together into a single batch on the CPU before being handed over to the `pops`.

    """
    device: torch.device

    def __init__(
        self,
        ops: Sequence[Union[None, TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Sequence[Union[None, NumpyOp,
                                                                                Scheduler[NumpyOp]]]] = None,
        slicers: Union[None, Slicer, Scheduler[Slicer], Sequence[Union[None, Slicer, Scheduler[Slicer]]]] = None,
    ) -> None:
        super().__init__(target_type='torch',
                         device=get_device(),
                         ops=ops,
                         postprocessing=postprocessing,
                         slicers=slicers)

    def __call__(self,
                 mode: str,
                 epoch: int,
                 ds_id: str,
                 desired_output_keys: Optional[Set[str]] = None,
                 warmup: bool = False,
                 eager: bool = False) -> Self:
        super().__call__(mode=mode,
                         epoch=epoch,
                         ds_id=ds_id,
                         desired_output_keys=desired_output_keys,
                         warmup=warmup,
                         eager=eager)
        if self.device.type != "cpu":
            for model in self.ctx_models:
                # move model variables to gpu
                model.to(self.device)
                if model.current_optimizer and mode == "train":
                    # move optimizer variables to gpu
                    self._move_optimizer_between_device(model.current_optimizer.state, self.device)
        # Set all of the contiguous final updates to defer their updates by default to enable things like CycleGan
        # This is not necessary for TF because overriding tf weights does not confuse the gradient tape computation
        for op in reversed(self.ctx_ops):
            if isinstance(op, UpdateOp):
                op._old_defer = op.defer
                op.defer = True
            else:
                break
        return self

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
                except (RuntimeError, AssertionError, AttributeError):
                    pass

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        """Clean up the network after running an epoch.

        In this case we move all of the models from the GPU(s) back to the CPU.
        """
        if self.device.type != "cpu":
            for model in self.ctx_models:
                # move model variables to cpu
                model.to("cpu")
                if model.current_optimizer and self.ctx_state["mode"] == "train":
                    # move optimizer variables to cpu
                    self._move_optimizer_between_device(model.current_optimizer.state, "cpu")
        # Set the final update ops back to their original defer status
        for op in reversed(self.ctx_ops):
            if isinstance(op, UpdateOp):
                op.defer = op.__dict__.get('_old_defer', op.defer)
            else:
                break
        super().__exit__(*exc)

    def _get_effective_batch_input(self, batch: MutableMapping[str, Any]) -> Dict[str, Any]:
        """Copy input data from the the CPU onto the GPU(s).

        This method will filter inputs from the batch so that only data required by the network during execution will be
        copied to the GPU.

        Args:
            batch: The input data to be moved.

        Returns:
            The input data ready for use on GPU(s).
        """
        if self.device.type != "cpu":
            new_batch = {
                key: move_tensors_to_device(batch[key], self.device)
                for key in self.ctx_gpu_inputs if key in batch
            }
        else:
            new_batch = {key: batch[key] for key in self.ctx_gpu_inputs if key in batch}
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
        batch_in = self._get_effective_batch_input(batch)
        self.ctx_state["tape"] = NonContext()
        # gpu operation
        with torch.no_grad() if not self.ctx_state["req_grad"] else NonContext():
            with torch.autocast(device_type=self.device.type) if self.mixed_precision else NonContext():
                self._forward_batch(batch_in, self.ctx_state, self.ctx_ops)

        # copy data to cpu
        if self.device.type != "cpu":
            prediction = {
                key: move_tensors_to_device(detach_tensors(batch_in[key]), "cpu")
                for key in self.ctx_outputs if key in batch_in
            }
        else:
            prediction = {key: detach_tensors(batch_in[key]) for key in self.ctx_outputs if key in batch_in}
        return batch, prediction


@traceable(blacklist=('ctx_lock', ))
class TFNetwork(BaseNetwork):
    """An extension of BaseNetwork for TensorFlow models.

    Args:
        ops: The ops defining the execution graph for this Network.
        postprocessing: A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed.
            Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points.
        slicers: Slicers to use if you want to cut apart a single batch of data into multiple slices in order to fit
            them onto a smaller GPU. After cutting the data apart and running through the `ops`, the samples are fused
            back together into a single batch on the CPU before being handed over to the `pops`.
    """
    def __init__(
        self,
        ops: Sequence[Union[None, TensorOp, Scheduler[TensorOp]]],
        postprocessing: Union[None, NumpyOp, Scheduler[NumpyOp], Sequence[Union[None, NumpyOp,
                                                                                Scheduler[NumpyOp]]]] = None,
        slicers: Union[None, Slicer, Scheduler[Slicer], Sequence[Union[None, Slicer, Scheduler[Slicer]]]] = None,
    ) -> None:
        super().__init__(target_type='tf', device=None, ops=ops, postprocessing=postprocessing, slicers=slicers)

    def __call__(self,
                 mode: str,
                 epoch: int,
                 ds_id: str,
                 desired_output_keys: Optional[Set[str]] = None,
                 warmup: bool = False,
                 eager: bool = False) -> Self:
        super().__call__(mode=mode,
                         epoch=epoch,
                         ds_id=ds_id,
                         desired_output_keys=desired_output_keys,
                         warmup=warmup,
                         eager=eager)
        # Don't cause a re-trace just because epoch changed
        self.ctx_state["epoch"] = tf.convert_to_tensor(self.ctx_state["epoch"])
        # Need to re-trace the TF graph if optimizer or layer trainable setting is changing:
        trainable_str = "".join([str(layer.trainable) for model in self.ctx_models for layer in model.layers])
        opt_str = "x".join(
            [str(id(model.current_optimizer)) for model in self.ctx_models if hasattr(model, 'current_optimizer')])
        self.ctx_state["_force_tf_retrace"] = hash(trainable_str + opt_str)  # Hash to keep at fixed memory overhead
        self.ctx_manual_gpu_data_handling = False
        return self

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        # This prevents a tf graph memory leak that would slow down long trainings. Since we
        # re-build graphs every epoch there is no reason to keep old ones around.
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            pass  # TODO - Find a way to clear graph for multi-gpu
        else:
            tf.keras.backend.clear_session()
        super().__exit__(*exc)

    def _run_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a forward step through the Network on a batch of data.

        Implementations of this method within derived classes should handle bringing the prediction data back from the
        (multi-)GPU environment to the CPU. This method expects that Network.load_epoch() has already been invoked.

        Args:
            batch: The batch of data serving as input to the Network.

        Returns:
            (batch_data, prediction_data)
        """
        batch_in = self._get_effective_batch_input(batch)
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            with Suppressor(allow_pyprint=True, show_if_exception=True):
                if self.ctx_manual_gpu_data_handling:
                    batch_in = next(
                        iter(strategy.experimental_distribute_dataset(tf.data.Dataset.from_tensors(batch_in))))
                if self.ctx_state["eager"]:
                    prediction = strategy.run(self._forward_step_eager,
                                              args=(batch_in, self.ctx_state, self.ctx_ops, to_list(self.ctx_outputs)))
                else:
                    prediction = strategy.run(self._forward_step_static,
                                              args=(batch_in, self.ctx_state, self.ctx_ops, to_list(self.ctx_outputs)))
            batch = self._per_replica_to_global(batch)
            prediction = self._per_replica_to_global(prediction)
        else:
            with Suppressor(allow_pyprint=True, show_if_exception=True):
                if self.ctx_state["eager"]:
                    prediction = self._forward_step_eager(batch_in,
                                                          self.ctx_state,
                                                          self.ctx_ops,
                                                          to_list(self.ctx_outputs))
                else:
                    prediction = self._forward_step_static(batch_in,
                                                           self.ctx_state,
                                                           self.ctx_ops,
                                                           to_list(self.ctx_outputs))
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

    def _get_effective_batch_input(self, batch: MutableMapping[str, Any]) -> Dict[str, Any]:
        """Filter input data so that only the data required by the Network is moved onto the GPU.

        Args:
            batch: An unfiltered batch of input data.

        Returns:
            The filtered input data ready for use on GPU(s).
        """
        new_batch = {}
        for key in self.ctx_gpu_inputs:
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

    @tf.function(reduce_retracing=True)
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

    def _do_transform(self, batch: Dict[str, Array]) -> Dict[str, Array]:
        # Distribute multi-gpu data for processing
        sub_sample = False
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            batch_size, num_devices = get_batch_size(batch), strategy.num_replicas_in_sync
            if batch_size < num_devices:
                batch = self._fill_batch(batch, num_devices - batch_size)
                sub_sample = True
            self.ctx_manual_gpu_data_handling = True
        batch = super()._do_transform(batch)
        if sub_sample:
            batch = self._subsample_data(batch, batch_size)
            self.ctx_manual_gpu_data_handling = False
        return batch

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
            return tf.pad(data, paddings=paddings, mode="symmetric")
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
        if isinstance(data, (dict, ChainMap)):
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


@overload
def build(model_fn: Callable[[], Model],
          optimizer_fn: Union[None, str, Scheduler, Callable],
          weights_path: Union[str, None, List[Union[str, None]]] = None,
          model_name: Union[str, List[str], None] = None,
          mixed_precision: bool = False) -> Model:
    ...


@overload
def build(model_fn: Callable[[], Sequence[Model]],
          optimizer_fn: Union[None, str, Scheduler, Callable, Sequence[Union[None, str, Callable, Scheduler]]],
          weights_path: Union[str, None, List[Union[str, None]]] = None,
          model_name: Union[str, List[str], None] = None,
          mixed_precision: bool = False) -> List[Model]:
    ...


def build(model_fn: Callable[[], Union[Model, Sequence[Model]]],
          optimizer_fn: Union[None, str, Scheduler, Callable, Sequence[Union[None, str, Callable, Scheduler]]],
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
        names = [
            "model" if i + fe.fe_build_count == 0 else "model{}".format(i + fe.fe_build_count) for i in range(num_names)
        ]
        fe.fe_build_count += num_names
        return names

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
        if sys.platform == 'darwin':
            warn("Mixed Precision is not currently supported on Mac / Metal. This flag will be ignored.")
            mixed_precision = False
        else:
            mixed_precision_tf.set_global_policy(mixed_precision_tf.Policy('mixed_float16'))
    else:
        mixed_precision_tf.set_global_policy(mixed_precision_tf.Policy('float32'))
    models = None
    if get_num_gpus() > 1:
        # We need to figure out whether model_fn returns tf models or torch models
        if not isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy):
            # If we've already done this and gotten TF model, the above flag will be set and this will be skipped. If we
            # are dealing with pytorch models, the model_fn() invocation will be kept so as to not waste clock cycles.
            models = to_list(model_fn())
            if isinstance(models[0], tf.keras.Model):
                models = None  # We will re-instantiate the models again now that we know we need MirroredStrategy
                tf.keras.backend.clear_session()  # This will reset the automatic layer naming in case user is
                # extracting intermediate layer outputs by name
                tf.distribute.experimental_set_strategy(tf.distribute.MirroredStrategy())
    models, optimizer_fn = to_list(model_fn()) if models is None else models, to_list(optimizer_fn)
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
    elif isinstance(model, Sequential):
        raise DeprecationWarning("Importing from tensorflow.python.keras.models/layers is deprecated. Import from "
                                 "tensorflow.keras.models/layers instead")
    else:
        raise ValueError("unrecognized model format: {}".format(type(model)))
    # torch multi-gpu handling
    if framework == "torch" and get_num_gpus() > 1:
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


def _build_optimizer(
    optimizer_fn: Union[str, Callable, None], model: Model, framework: str, mixed_precision: bool
) -> Union[None, tf.optimizers.Optimizer, tf.optimizers.legacy.Optimizer, torch.optim.Optimizer]:
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
    # The legacy optimizers appear to be faster than the new ones on both mac and linux. Revisit this speed test again
    # after tf version > 2.12
    tf_optimizer_fn = {
        'adadelta': tf.optimizers.legacy.Adadelta,
        'adagrad': tf.optimizers.legacy.Adagrad,
        'adam': tf.optimizers.legacy.Adam,
        'adamax': tf.optimizers.legacy.Adamax,
        'rmsprop': tf.optimizers.legacy.RMSprop,
        'sgd': tf.optimizers.legacy.SGD
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


def _optimizer_fn_to_optimizer(
        optimizer_fn: Union[Callable, None], model: Model, framework: str,
        mixed_precision: bool) -> Union[None, tf.optimizers.legacy.Optimizer, torch.optim.Optimizer]:
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
            if sys.platform == 'darwin' and hasattr(optimizer, 'jit_compile'):
                # Mac doesn't support XLA acceleration as of TF 2.11
                # TODO - check compatibility again in future release
                global _MAC_BUILD_WARNING
                if not _MAC_BUILD_WARNING:
                    warn("JIT compiling of optimizers is not currently supported by MacOS and will be disabled. You "
                         "may want to use an optimizer from tf.optimizers.legacy instead for better speed.")
                    _MAC_BUILD_WARNING = True
                optimizer.jit_compile = False
            # initialize optimizer variables
            if hasattr(optimizer, 'build'):
                optimizer.build(var_list=model.trainable_variables)
            else:
                _ = optimizer.iterations
                if hasattr(optimizer, '_create_hypers'):
                    optimizer._create_hypers()
                if hasattr(optimizer, '_create_slots'):
                    optimizer._create_slots(model.trainable_variables)
            assert isinstance(optimizer, (tf.optimizers.Optimizer, tf.keras.optimizers.experimental.Optimizer,
                                          tf.optimizers.legacy.Optimizer)), \
                f"optimizer_fn should generate tensorflow optimizer, but got {type(optimizer)}"
            # handle mixed precision loss scaling
            if mixed_precision:
                if isinstance(optimizer, tf.optimizers.legacy.Optimizer):
                    optimizer = LossScaleOptimizer(optimizer)
                else:
                    optimizer = LossScaleOptimizerV3(optimizer)
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
