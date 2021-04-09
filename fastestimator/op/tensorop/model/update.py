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
from typing import Any, Dict, Iterable, List, Optional, Set, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.get_gradient import get_gradient
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.update_model import update_model
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_set

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the loss.

    Args:
        model: Model instance compiled by fe.build.
        loss_name: The input loss key.
        gradients: The gradients key which the model will update according to. The provided gradients will be directly
            used for model update. If it is None, the gradients will be computed from the input key of `loss_name`, and
            this will take care of the scaling for mixed-precision training. `Gradients` should be None when the model
            is mixed-precision.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        merge_grad: The gradient accumulation times before model update. Ex: if `merge_grad` = 3, for every three Op
            calls only the third one updates the model. The first two calls only accumulate its gradients. This default
            value is one 1 and it will update the model at every step.
        defer: Whether to defer the actual application of the update until the end of the step. This can be necessary
            in PyTorch when trying to update multiple models which depend on one another (ex. certain GANs). By default,
            all UpdateOps which appear contiguously as the last ops of a Network will be deferred. We hope that you will
            never need to worry about this flag, but it's here for you if you need it.

    Raise:
        ValueError: When model is mixed-precision and `gradients` is provided.
        ValueError: network framework is not one of "tf" or "torch".
        RuntimeError: If attempting to modify a PyTorch model which relied on gradients within a different PyTorch model
            which has in turn already undergone a non-deferred update.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 loss_name: str,
                 gradients: Optional[str] = None,
                 mode: Union[None, str, Iterable[str]] = "train",
                 merge_grad: int = 1,
                 defer: bool = False):
        if gradients is None:
            super().__init__(inputs=loss_name, outputs=None, mode=mode)
        elif model.mixed_precision:
            raise ValueError("Mixed precision training cannot take input gradients, because the gradient need to be "
                             "computed in this module")
        else:
            super().__init__(inputs=gradients, outputs=None, mode=mode)

        if not hasattr(model, "loss_name"):
            model.loss_name = {loss_name}
        else:
            model.loss_name.add(loss_name)

        self.model = model
        self.retain_graph = False
        self.weight_decay = isinstance(self.model, tf.keras.Model) and self.model.losses
        self.defer = defer
        self.gradients = gradients
        self.loss_name = loss_name
        self.merge_grad = merge_grad
        self.framework = None

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        self.framework = framework
        if self.merge_grad > 1:
            if framework == "tf":
                self.step = tf.Variable(0, trainable=False)
                self.grad_sum = [tf.Variable(tf.zeros_like(x), trainable=False) for x in self.model.trainable_variables]
            elif framework == "torch":
                self.step = torch.tensor(0).to(device)
                self.grad_sum = [torch.zeros_like(x).to(device) for x in self.model.parameters() if x.requires_grad]

    def get_fe_models(self) -> Set[Model]:
        return {self.model}

    def get_fe_loss_keys(self) -> Set[str]:
        return to_set(self.loss_name)

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> None:
        if state["warmup"]:
            return

        if self.gradients is None:  # data is loss
            loss = self._loss_preprocess(data, state)
            gradients = self._get_gradient(loss, state)
            gradients = self._gradient_postprocess(gradients, state)

        else:  # data is gradients
            gradients = data

        if self.merge_grad > 1:
            self._merge_grad_update(gradients, state)
        else:
            update_model(model=self.model,
                         gradients=gradients,
                         tape=state["tape"],
                         defer=self.defer,
                         deferred=state["deferred"])

    def _loss_preprocess(self, loss, state):
        """Loss preprocess for multi-GPU and mixed-precision training
        """
        if self.weight_decay:
            loss = loss + tf.reduce_sum(self.model.losses)
        loss = reduce_mean(loss)

        if self.framework == "tf":
            # scale up loss for mixed precision training to avoid underflow
            if self.model.mixed_precision:
                loss = self.model.current_optimizer.get_scaled_loss(loss)
            # for multi-gpu training, the gradient will be combined by sum, normalize the loss
            strategy = tf.distribute.get_strategy()
            if isinstance(strategy, tf.distribute.MirroredStrategy):
                loss = loss / strategy.num_replicas_in_sync

        elif self.framework == "torch":
            # scale up loss for mixed precision training to avoid underflow
            if self.model.current_optimizer.scaler is not None:
                loss = self.model.current_optimizer.scaler.scale(loss)

        else:
            raise ValueError(f"Unrecognized framework {self.framework}")

        return loss

    def _get_gradient(self, loss, state):
        if self.framework == "tf":
            gradients = get_gradient(loss, self.model.trainable_variables, tape=state['tape'])

        elif self.framework == "torch":
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            try:
                gradients = get_gradient(loss, trainable_params, retain_graph=self.retain_graph)
            except RuntimeError as err:
                if err.args and isinstance(err.args[0], str) and err.args[0].startswith(
                        'one of the variables needed for gradient computation has been modified by an inplace operation'
                ):
                    raise RuntimeError(
                        "When computing gradients for '{}', some variables it relied on during the forward pass had"
                        " been updated. Consider setting defer=True in earlier UpdateOps related to models which "
                        "interact with this one.".format(self.model.model_name))
                raise err

        else:
            raise ValueError(f"Unrecognized framework {self.framework}")

        return gradients

    def _gradient_postprocess(self, gradients, state):
        """ Gradient postprocess for multi-GPU and mixed-precision training
        """
        if self.framework == "tf":
            if self.gradients is not None:  # user provide gradients
                strategy = tf.distribute.get_strategy()  # need tape.stop_recording() ?
                # for multi-gpu training, the gradient will be combined by sum, normalize the gradient
                if isinstance(strategy, tf.distribute.MirroredStrategy):
                    gradients = gradients / strategy.num_replicas_in_sync

            # scale down gradient to balance scale-up loss
            if self.model.mixed_precision:
                with state["tape"].stop_recording():
                    gradients = self.model.current_optimizer.get_unscaled_gradients(gradients)

        elif self.framework == "torch":
            pass

        else:
            raise ValueError(f"Unrecognized framework {self.framework}")

        return gradients

    def _merge_grad_update(self, gradients, state):
        current_grad = gradients
        if self.framework == "tf":
            self.step.assign_add(1)
            # add current gradient to the accumulated gradient
            for gs, g in zip(self.grad_sum, current_grad):
                gs.assign_add(g)
            if self.step % self.merge_grad == 0:
                average_grad = [gs / self.merge_grad for gs in self.grad_sum]
                # apply update with cumulative gradient
                update_model(model=self.model,
                             gradients=average_grad,
                             tape=state["tape"],
                             defer=self.defer,
                             deferred=state["deferred"])

                # zero-out cumulative gradient
                for gs in self.grad_sum:
                    gs.assign_sub(gs)

        elif self.framework == "torch":
            self.step += 1
            # add current gradient to the accumulated gradient
            for gs, g in zip(self.grad_sum, current_grad):
                gs += g
            if self.step % self.merge_grad == 0:
                average_grad = [gs / self.merge_grad for gs in self.grad_sum]
                # apply update with cumulative gradient
                update_model(model=self.model, gradients=average_grad, defer=self.defer, deferred=state["deferred"])

                # zero-out cumulative gradient
                for gs in self.grad_sum:
                    gs -= gs
        else:
            raise ValueError(f"Unrecognized framework {self.framework}")
