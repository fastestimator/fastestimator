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

import torch

from fastestimator.backend._get_gradient import get_gradient
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._update_model import update_model
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.base_util import to_set, warn
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import get_num_gpus


@traceable()
class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the loss.

    Args:
        model: Model instance compiled by fe.build.
        loss_name: The input loss key.
        gradients: An optional key containing model gradients. These will be directly applied to the model weights
            during an update. If not provided, gradients will be computed based on the specified loss_name, which will
            automatically handle any desired mixed-precision scaling. This argument shouldn't be used if mixed-precision
            training is enabled.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        merge_grad: The gradient accumulation times before model update. Ex: if `merge_grad` = 3, for every three Op
            calls only the third one updates the model. The first two calls only accumulate its gradients. This default
            value is 1 and it will update the model at every step.
        defer: Whether to defer the actual application of the update until the end of the step. This can be necessary
            in PyTorch when trying to update multiple models which depend on one another (ex. certain GANs). By default,
            all UpdateOps which appear contiguously as the last ops of a Network will be deferred. We hope that you will
            never need to worry about this flag, but it's here for you if you need it.

    Raise:
        ValueError: When model is mixed-precision and `gradients` is provided.
        ValueError: Network framework is not "torch".
        ValueError: `merge_grad` is larger than 1 in multi-GPU configuration.
        RuntimeError: If attempting to modify a PyTorch model which relied on gradients within a different PyTorch model
            which has in turn already undergone a non-deferred update.
    """
    _old_defer: bool  # Used by the Network to automagically fix defer values

    def __init__(self,
                 model: torch.nn.Module,
                 loss_name: str,
                 gradients: Optional[str] = None,
                 mode: Union[None, str, Iterable[str]] = "train",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 merge_grad: int = 1,
                 defer: bool = False):
        if gradients is None:
            super().__init__(inputs=loss_name, outputs=None, mode=mode, ds_id=ds_id)
        else:
            if model.mixed_precision:
                raise ValueError("Mixed precision training cannot take input gradients, because the gradients need to "
                                 "be computed in this module")
            super().__init__(inputs=gradients, outputs=None, mode=mode, ds_id=ds_id)

        if get_num_gpus() > 1 and merge_grad > 1:
            raise ValueError("Currently FastEstimator doesn't support merge_grad feature in multi-GPU configuration "
                             "and thus 'merge_grad' cannot be larger than 1")

        if not hasattr(model, "loss_name"):
            model.loss_name = {loss_name}
        else:
            model.loss_name.add(loss_name)

        self.model = model
        self.retain_graph = False
        self.defer = defer
        self.gradients = gradients
        self.loss_name = loss_name
        self.merge_grad = merge_grad
        self.framework = None

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if framework not in ["torch"]:
            raise ValueError(f"Unrecognized framework {framework}")

        self.framework = framework

        if self.merge_grad > 1:
            self.step = torch.tensor(0, dtype=torch.int64).to(device)
            self.grad_sum = [torch.zeros_like(x).to(device) for x in self.model.parameters() if x.requires_grad]

    def get_fe_models(self) -> Set[torch.nn.Module]:
        return {self.model}

    def get_fe_loss_keys(self) -> Set[str]:
        return to_set(self.loss_name)

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def forward(self, data: Union[torch.Tensor, List[torch.Tensor]], state: Dict[str, Any]) -> None:
        if state["warmup"]:
            return

        if self.gradients is None:  # data is loss
            loss = self._loss_preprocess(data)
            gradients = self._get_gradient(loss)
        else:  # data is gradients
            gradients = data
        gradients = self._gradient_postprocess(gradients)

        if self.merge_grad > 1:
            self._merge_grad_update(gradients, deferred=state["deferred"])
        else:
            update_model(model=self.model, gradients=gradients, defer=self.defer, deferred=state["deferred"])

    def _loss_preprocess(self, loss: Union[torch.Tensor,
                                           List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Loss preprocess for multi-GPU and mixed-precision training.

        Args:
            loss: Unprocessed loss.

        Returns:
            Processed loss.
        """
        loss = reduce_mean(loss)
        if self.model.current_optimizer.scaler is not None:
            # scale up loss for mixed precision training to avoid underflow
            loss = self.model.current_optimizer.scaler.scale(loss)
        return loss

    def _get_gradient(self, loss: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get gradient from loss with repect to self.model.

        Args:
            loss: Input loss.

        Returns:
            Computed gradients.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            gradients = get_gradient(loss, trainable_params, retain_graph=self.retain_graph)
        except RuntimeError as err:
            if err.args and isinstance(err.args[0], str) and err.args[0].startswith(
                    'one of the variables needed for gradient computation has been modified by an inplace operation'):
                raise RuntimeError(
                    "When computing gradients for '{}', some variables it relied on during the forward pass had"
                    " been updated. Consider setting defer=True in earlier UpdateOps related to models which "
                    "interact with this one.".format(self.model.model_name))
            raise err

        return gradients

    def _gradient_postprocess(
            self, gradients: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Gradient postprocess for multi-GPU and mixed-precision training.

        Args:
            gradients: Unprocessed gradients.

        Returns:
            Processed gradients.
        """
        return gradients

    def _merge_grad_update(self,
                           gradients: Union[torch.Tensor, List[torch.Tensor]],
                           deferred: Optional[Dict[str, List[Callable[[], None]]]] = None) -> None:
        """Accumulate gradients and update the model at certain frequency of invocation.

        Args:
            gradients: Input gradients.
            deferred: A dictionary in which model update functions are stored.
        """

        # add current gradient to the cumulative gradient
        for gs, g in zip(self.grad_sum, gradients):
            self._assign_add(gs, g)

        self._assign_add(self.step, 1)

        if self.step % self.merge_grad == 0:
            average_grad = [gs / self.merge_grad for gs in self.grad_sum]
            update_model(model=self.model, gradients=average_grad, defer=self.defer, deferred=deferred)
            for gs in self.grad_sum:
                self._assign_add(gs, -gs)  # zero the gradient in place

    def _assign_add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """In-place addition for PyTorch. `a` = `a` + `b`

        Args:
            a: A tensor where in-place addition happens.
            b: Amount to be added.
        """
        a += b
