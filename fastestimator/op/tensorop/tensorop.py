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
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.op.op import Op
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class TensorOp(Op):
    """An Operator class which takes and returns tensor data.

    These Operators are used in fe.Network to perform graph-based operations like neural network training.
    """
    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        """A method which will be invoked in order to transform data.

        This method will be invoked on batches of data.

        Args:
            data: The batch from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data
            dictionary based on whatever keys this Op declares as its `outputs`.
        """
        return data

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        """A method which will be invoked during Network instantiation.

        This method can be used to augment the natural __init__ method of the TensorOp once the desired backend
        framework is known.

        Args:
            framework: Which framework this Op will be executing in. One of 'tf' or 'torch'.
            device: Which device this Op will execute on. Usually 'cuda:0' or 'cpu'. Only populated when the `framework`
                is 'torch'.
        """
        pass

    # ###########################################################################
    # The methods below this point can be ignored by most non-FE developers
    # ###########################################################################

    # noinspection PyMethodMayBeStatic
    def get_fe_models(self) -> Set[Model]:
        """A method to get any models held by this Op.

        All users and most developers can safely ignore this method. This method may be invoked to gather and manipulate
        models, for example by the Network during load_epoch().

        Returns:
            Any models held by this Op.
        """
        return set()

    # noinspection PyMethodMayBeStatic
    def get_fe_loss_keys(self) -> Set[str]:
        """A method to get any loss keys held by this Op.

        All users and most developers can safely ignore this method. This method may be invoked to gather information
        about losses, for example by the Network in get_loss_keys().

        Returns:
            Any loss keys held by this Op.
        """
        return set()

    # noinspection PyMethodMayBeStatic
    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        """A method to get / set whether this Op should retain network gradients after computing them.

        All users and most developers can safely ignore this method. Ops which do not compute gradients should leave
        this method alone. If this method is invoked with `retain` as True or False, then the gradient computations
        performed by this Op should retain or discard the graph respectively afterwards.

        Args:
            retain: If None, then return the current retain_graph status of the Op. If True or False, then set the
                retain_graph status of the op to the new status and return the new status.

        Returns:
            Whether this Op will retain the backward gradient graph after it's forward pass, or None if this Op does not
            compute backward gradients.
        """
        return None


@traceable()
class LambdaOp(TensorOp):
    """An Operator that performs any specified function as forward function.

    Args:
        fn: The function to be executed.
        inputs: Key(s) from which to retrieve data from the data dictionary.
        outputs: Key(s) under which to write the outputs of this Op back to the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 fn: Callable,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.fn = fn
        self.in_list = True

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        return self.fn(*data)
