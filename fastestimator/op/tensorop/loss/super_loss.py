# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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
from math import e
from typing import Any, Dict, List, Optional, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.exp import exp
from fastestimator.backend.lambertw import lambertw
from fastestimator.backend.maximum import maximum
from fastestimator.backend.ones_like import ones_like
from fastestimator.backend.pow import pow
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.util import to_number

Tensor = TypeVar('Tensor', tf.Tensor, tf.Variable, torch.Tensor)


class SuperLoss(LossOp):
    """Loss class to compute a 'super loss' (automatic curriculum learning) based on a regular loss.

    This class adds automatic curriculum learning on top of any other loss metric. It is especially useful in for noisy
    datasets. See https://papers.nips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf for details.

    Args:
        loss: A loss object which we use to calculate the underlying regular loss. This should be an object of type
            fe.op.tensorop.loss.loss.LossOp.
        threshold: Either a constant value corresponding to an average expected loss (for example log(n_classes) for
            cross-entropy classification), or 'exp' to use an exponential moving average loss.
        regularization: The regularization parameter to use for the super loss (must by >0, as regularization approaches
            infinity the SuperLoss converges to the regular loss value).
        average_loss: Whether the final loss should be averaged or not.
        output_confidence: If not None then the confidence scores for each sample will be written into the specified
            key. This can be useful for finding difficult or mislabeled data.

    Raises:
        ValueError: If the provided `loss` has multiple outputs or the `regularization` / `threshold` parameters are
            invalid.
    """
    def __init__(self,
                 loss: LossOp,
                 threshold: Union[float, str] = 'exp',
                 regularization: float = 1.0,
                 average_loss: bool = True,
                 output_confidence: Optional[str] = None):
        if len(loss.outputs) != 1 or loss.out_list:
            raise ValueError("SuperLoss only supports lossOps which have a single output.")
        self.loss = loss
        self.loss.average_loss = False
        super().__init__(inputs=loss.inputs,
                         outputs=loss.outputs[0] if not output_confidence else (loss.outputs[0], output_confidence),
                         mode=loss.mode,
                         average_loss=average_loss)
        if not isinstance(threshold, str):
            threshold = to_number(threshold).item()
        if not isinstance(threshold, float) and threshold != 'exp':
            raise ValueError(f'SuperLoss threshold parameter must be "exp" or a float, but got {threshold}')
        self.tau_method = threshold
        if regularization <= 0:
            raise ValueError(f"SuperLoss regularization parameter must be greater than 0, but got {regularization}")
        self.lam = regularization
        self.cap = -1.9999998 / e  # Slightly more than -2 / e for numerical stability
        self.initialized = {}
        self.tau = {}

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        self.loss.build(framework, device)
        if framework == 'tf':
            self.initialized = {
                'train': tf.Variable(False, trainable=False),
                'eval': tf.Variable(False, trainable=False),
                'test': tf.Variable(False, trainable=False),
                'infer': tf.Variable(False, trainable=False)
            }
            if self.tau_method == 'exp':
                self.tau = {
                    'train': tf.Variable(0.0, trainable=False),
                    'eval': tf.Variable(0.0, trainable=False),
                    'test': tf.Variable(0.0, trainable=False),
                    'infer': tf.Variable(0.0, trainable=False)
                }
            else:
                self.tau = {
                    'train': tf.Variable(self.tau_method, trainable=False),
                    'eval': tf.Variable(self.tau_method, trainable=False),
                    'test': tf.Variable(self.tau_method, trainable=False),
                    'infer': tf.Variable(self.tau_method, trainable=False)
                }
            self.cap = tf.constant(self.cap)
        elif framework == 'torch':
            self.initialized = {
                'train': torch.tensor(False).to(device),
                'eval': torch.tensor(False).to(device),
                'test': torch.tensor(False).to(device),
                'infer': torch.tensor(False).to(device)
            }
            if self.tau_method == 'exp':
                self.tau = {
                    'train': torch.tensor(0.0).to(device),
                    'eval': torch.tensor(0.0).to(device),
                    'test': torch.tensor(0.0).to(device),
                    'infer': torch.tensor(0.0).to(device)
                }
            else:
                self.tau = {
                    'train': torch.tensor(self.tau_method).to(device),
                    'eval': torch.tensor(self.tau_method).to(device),
                    'test': torch.tensor(self.tau_method).to(device),
                    'infer': torch.tensor(self.tau_method).to(device)
                }
            self.cap = torch.tensor(self.cap).to(device)
            self.lam = torch.tensor(self.lam).to(device)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        base_loss = self.loss.forward(data, state)
        tau = self._accumulate_tau(base_loss, state['mode'], state['warmup'])
        beta = (base_loss - tau) / self.lam
        # TODO The authors say to remove the gradients. Need to check whether this is necessary (speed or metrics)
        ln_sigma = -lambertw(0.5 * maximum(self.cap, beta))
        super_loss = (base_loss - tau) * exp(ln_sigma) + self.lam * pow(ln_sigma, 2)

        if self.average_loss:
            super_loss = reduce_mean(super_loss)

        if len(self.outputs) == 2:
            # User requested that the confidence score be returned
            return [super_loss, exp(ln_sigma)]

        return super_loss

    def _accumulate_tau(self, loss: Tensor, mode: str, warmup: bool) -> Tensor:
        """Determine an average loss value based on a particular method chosen during __init__.

        Right now this only supports constant values or exponential averaging. The original paper also proposed global
        averaging, but they didn't find much difference between the three methods and global averaging would more
        complicated memory requirements.

        Args:
            loss: The current step loss.
            mode: The current step mode.
            warmup: Whether running in warmup mode or not.

        Returns:
            Either the static value provided at __init__, or an exponential moving average of the loss over time.
        """
        if self.tau_method == 'exp':
            if _read_variable(self.initialized[mode]):
                _assign(self.tau[mode], self.tau[mode] - 0.1 * (self.tau[mode] - reduce_mean(loss)))
            else:
                _assign(self.tau[mode], reduce_mean(loss))
                if not warmup:
                    _assign(self.initialized[mode], ones_like(self.initialized[mode]))
        return self.tau[mode]


def _read_variable(variable: Tensor) -> Tensor:
    """Read a variable.

    For some unknown reason, tf.Variable(False) on a multi-gpu machine will evaluate as True during an if-check, so need
    to invoke the read method manually.

    Args:
        variable: The variable to be read.

    Returns:
        The `variable` value.
    """
    if isinstance(variable, torch.Tensor):
        return variable
    return variable.read_value()


def _assign(variable: Tensor, value: Tensor) -> None:
    """In place assignment of `value` to a `variable`.

    Args:
        variable: The tensor to be modified.
        value: The new value to be inserted into the `variable`.
    """
    if isinstance(variable, torch.Tensor):
        variable.copy_(value.detach())
    else:
        variable.assign(value)
