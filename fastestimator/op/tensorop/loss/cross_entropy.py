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
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend._binary_crossentropy import binary_crossentropy
from fastestimator.backend._categorical_crossentropy import categorical_crossentropy
from fastestimator.backend._sparse_categorical_crossentropy import sparse_categorical_crossentropy
from fastestimator.op.tensorop.loss.loss import LossOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class CrossEntropy(LossOp):
    """Calculate Element-Wise CrossEntropy (binary, categorical or sparse categorical).

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>].
        outputs: String key under which to store the computed loss value.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        from_logits: Whether y_pred is logits (without softmax).
        average_loss: Whether to average the element-wise loss after the Loss Op.
        form: What form of cross entropy should be performed ('binary', 'categorical', 'sparse', or None). None will
            automatically infer the correct form based on tensor shape: if the both y_pred and y_true are rank-2 tensors
            then 'categorical' will be used, if y_pred is rank-2 tensors but y_true is rank-1 tensor, then `sparse` will
            be chosen, otherwise `binary` will be applied.
        class_weights: Dictionary mapping class indices to a weight for weighting the loss function. Useful when you
            need to pay more attention to samples from an under-represented class.

    Raises:
        AssertionError: If `class_weights` or it's keys and values are of unacceptable data types.
    """
    def __init__(self,
                 inputs: Union[Tuple[str, str], List[str]],
                 outputs: str,
                 mode: Union[None, str, Iterable[str]] = "!infer",
                 ds_id: Union[None, str, Iterable[str]] = None,
                 from_logits: bool = False,
                 average_loss: bool = True,
                 form: Optional[str] = None,
                 class_weights: Optional[Dict[int, float]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id, average_loss=average_loss)
        self.from_logits = from_logits
        self.form = form
        self.cross_entropy_fn = {
            "binary": binary_crossentropy,
            "categorical": categorical_crossentropy,
            "sparse": sparse_categorical_crossentropy
        }

        if class_weights:
            assert isinstance(class_weights, dict), \
                "class_weights should be a dictionary or have None value, got {}".format(type(class_weights))
            assert all(isinstance(key, int) for key in class_weights.keys()), \
                "Please ensure that the keys of the class_weight dictionary are of type: int"
            assert all(isinstance(value, float) for value in class_weights.values()), \
                "Please ensure that the values of the class_weight dictionary are of type: float"

        self.class_weights = class_weights
        self.class_dict = None

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        if self.class_weights:
            if framework == 'tf':
                keys_tensor = tf.constant(list(self.class_weights.keys()))
                vals_tensor = tf.constant(list(self.class_weights.values()))
                self.class_dict = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=1.0)
            elif framework == 'torch':
                self.class_dict = self.class_weights
            else:
                raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> Tensor:
        y_pred, y_true = data
        form = self.form
        if form is None:
            if len(y_pred.shape) == 2 and y_pred.shape[-1] > 1:
                if len(y_true.shape) == 2 and y_true.shape[-1] > 1:
                    form = "categorical"
                else:
                    form = "sparse"
            else:
                form = "binary"

        loss = self.cross_entropy_fn[form](y_pred,
                                           y_true,
                                           from_logits=self.from_logits,
                                           average_loss=self.average_loss,
                                           class_weights=self.class_dict)
        return loss
