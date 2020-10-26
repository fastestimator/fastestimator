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
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.network import BaseNetwork
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class Repeat(TensorOp):
    """Repeat a TensorOp several times in a row.

    Args:
        op: A TensorOp to be run one or more times in a row.
        repeat: How many times to repeat the `op`. This can also be a function return, in which case the function input
            names will be matched to keys in the data dictionary, and the `op` will be repeated until the function
            evaluates to False. The function evaluation will happen at the end of a forward call, so the `op` will
            always be evaluated at least once.

    Raises:
        ValueError: If `repeat` or `op` are invalid.
    """
    def __init__(self, op: TensorOp, repeat: Union[int, Callable[..., bool]] = 1) -> None:
        self.repeat_inputs = []
        extra_reqs = []
        if isinstance(repeat, int):
            if repeat < 1:
                raise ValueError(f"Repeat requires repeat to be >= 1, but got {repeat}")
        else:
            self.repeat_inputs.extend(inspect.signature(repeat).parameters.keys())
            extra_reqs = list(set(self.repeat_inputs) - set(op.outputs))
        self.repeat = repeat
        super().__init__(inputs=op.inputs + extra_reqs, outputs=op.outputs, mode=op.mode)
        self.ops = [op]
        self.retain_graph = None
        self.while_fn = None

    @property
    def op(self) -> TensorOp:
        return self.ops[0]

    def build(self, framework: str) -> None:
        self.op.build(framework)
        if framework == 'tf':
            self.while_fn = self._tf_while
        else:
            self.while_fn = self._torch_while

    def get_fe_models(self) -> Set[Model]:
        return self.op.get_fe_models()

    def get_fe_loss_keys(self) -> Set[str]:
        return self.op.get_fe_loss_keys()

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.op.fe_retain_graph(retain)

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'ops': [elem.__getstate__() if hasattr(elem, '__getstate__') else {} for elem in self.ops]}

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        # Set retain to true since might loop over a gradient aware op
        self.op.fe_retain_graph(True)

        data = {key: elem for key, elem in zip(self.inputs, data)}
        if isinstance(self.repeat, int):
            for i in range(self.repeat - 1):
                # Perform n-1 rounds with all ops having retain_graph == True
                BaseNetwork._forward_batch(data, state, self.ops)
            # Let retain be whatever it was meant to be for the final sequence
            self.op.fe_retain_graph(self.retain_graph)
            # Final round of ops
            BaseNetwork._forward_batch(data, state, self.ops)
        else:
            BaseNetwork._forward_batch(data, state, self.ops)
            data = self.while_fn(data, state)
            # TODO - Find some magic way to invoke this at the right moment
            self.op.fe_retain_graph(self.retain_graph)
        return [data[key] for key in self.outputs]

    def _torch_while(self, data: Dict[str, Tensor], state: Dict[str, Any]) -> Dict[str, Tensor]:
        """A helper function to invoke a while loop.

        Args:
            data: A data dictionary to be used during looping.
            state: The state variables to be considered during looping.

        Returns:
            A reference to the updated data dictionary.
        """
        while self.repeat(*[data[var_name] for var_name in self.repeat_inputs]):
            BaseNetwork._forward_batch(data, state, self.ops)
        return data

    def _tf_while(self, data: Dict[str, Tensor], state: Dict[str, Any]) -> Dict[str, Tensor]:
        """A helper function to invoke a while loop.

        Args:
            data: A data dictionary to be used during looping.
            state: The state variables to be considered during looping.

        Returns:
            A reference to the updated data dictionary.
        """
        args = ([data[var_name] for var_name in self.repeat_inputs], data, state)
        args = tf.while_loop(self._tf_cond, self._tf_body, args)
        return args[1]

    def _tf_cond(self, cnd: List[Tensor], data: Dict[str, Tensor], state: Dict[str, Any]) -> bool:
        """A helper function determine whether to keep invoking the while method.

        Note that `data` and `state` are unused here, but required since tf.while_loop needs the cond and body to have
        the same input argument signatures.

        Args:
            cnd: A list of arguments to be passed to the condition function.
            data: A data dictionary to be used during looping.
            state: The state variables to be considered during looping.

        Returns:
            Whether to continue looping.
        """
        return self.repeat(*cnd)

    def _tf_body(self, cnd: List[Tensor], data: Dict[str, Tensor],
                 state: Dict[str, Any]) -> Tuple[List[Tensor], Dict[str, Tensor], Dict[str, Any]]:
        """A helper function to execute the body of a while method.

        Note that `cnd` is unused here, but required since tf.while_loop needs the cond and body to have the same input
        argument signatures.

        Args:
            cnd: A list of arguments to be passed to the condition function.
            data: A data dictionary to be used during looping.
            state: The state variables to be considered during looping.

        Returns:
            The updated `cnd` values, along with the modified data and state dictionaries.
        """
        BaseNetwork._forward_batch(data, state, self.ops)
        return [data[var_name] for var_name in self.repeat_inputs], data, state
