#  Copyright 2022 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
from typing import Any, Iterable, List, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score

from fastestimator.trace.meta._per_ds import per_ds
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@per_ds
@traceable()
class AUCScore(Trace):
    """Compute Area Under the ROC Curve from prediction scores.

    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary. Array-like of shape
            (n_samples,) or (n_samples, n_classes).
            The binary and multiclass cases expect labels with shape (n_samples,)
            while the multilabel case expects binary label indicators with shape (n_samples, n_classes).
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary. Array-like of shape
            (n_samples,) or (n_samples, n_classes) Target scores/Probability estimates.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an
            argument like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in
            all ds_ids except for a particular one, you can pass an argument like "!ds1".
        output_name: Name of the key to store back to the state.
        average: {'micro', 'macro', 'samples', 'weighted'} or None,
            If ``None``, the scores for each class are returned. Otherwise, this determines the
            type of averaging performed on the data. Note: multiclass ROC AUC currently only handles the 'macro' and
            'weighted' averages. For multiclass targets, `average=None` is only implemented for `multi_class='ovo'`.
            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            ``'samples'``:
                Calculate metrics for each instance, and find their average.
            Will be ignored when ``y_true`` is binary.
        sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
        max_fpr: float > 0 and <= 1, default=None
            If not ``None``, the standardized partial AUC over the range [0, max_fpr] is returned.
            For the multiclass case, ``max_fpr``, should be either equal to ``None`` or ``1.0`` as AUC ROC partial
            computation currently is not supported for multiclass.
        multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
            Only used for multiclass targets. Determines the type of configuration to use. The default value raises
            an error, so either
            ``'ovr'`` or ``'ovo'`` must be passed explicitly.
            ``'ovr'``:
                Stands for One-vs-rest. Computes the AUC of each class against the rest. This
                treats the multiclass case in the same way as the multilabel case. Sensitive to class imbalance even
                when ``average == 'macro'``, because class imbalance affects the composition of each of the
                'rest' groupings.
            ``'ovo'``:
                Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes.
                Insensitive to class imbalance when ``average == 'macro'``.
        labels : array-like of shape (n_classes,), default=None
            Only used for multiclass targets. Used to get the missing labels which aren't present in input batch of
            ``y_true``.If ``None``, the numerical or sorted order of the labels in ``y_true`` is used.
        per_ds: Whether to automatically compute this metric individually for every ds_id it runs on, in addition to
            computing an aggregate across all ds_ids on which it runs. This is automatically False if `output_name`
            contains a "|" character.

        Raises:
            ValueError: Expected values of 'max_fpr' should be either None or between 0 and 1.
            ValueError: Expected values of 'multi_class' are ['raise', 'ovr' or 'ovo'].
            ValueError: Expected values of 'average' should be either None or ['micro', 'macro', 'samples', 'weighted'].
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[None, str, Iterable[str]] = ("eval", "test"),
                 ds_id: Union[None, str, Iterable[str]] = None,
                 output_name: str = "auc",
                 average: str = "macro",
                 sample_weight: Optional[List[float]] = None,
                 max_fpr: Optional[float] = None,
                 multi_class: str = 'raise',
                 labels: Optional[List[Any]] = None,
                 per_ds: bool = True):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode, ds_id=ds_id)
        self.per_ds = per_ds
        self.labels = labels
        self.sample_weight = sample_weight
        if max_fpr is not None and (max_fpr < 0 and max_fpr > 1):
            raise ValueError("Expected values of 'max_fpr' should be either None or between 0 and 1.")

        if multi_class not in ['raise', 'ovr', 'ovo']:
            raise ValueError("Expected values of 'multi_class' are ['raise', 'ovr' or 'ovo'].")

        if average is not None and average not in ['micro', 'macro', 'samples', 'weighted']:
            raise ValueError(
                "Expected values of 'average' should be either None or ['micro', 'macro', 'samples', 'weighted'].")

        self.max_fpr = max_fpr
        self.average = average
        self.multi_class = multi_class

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.y_true = np.array([])
        self.y_pred = np.array([])

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if self.y_pred.size == 0:
            self.y_pred = y_pred
            self.y_true = y_true
        else:
            self.y_pred = np.concatenate((self.y_pred, y_pred))
            self.y_true = np.concatenate((self.y_true, y_true))

    def get_auc(self):
        roc_auc = roc_auc_score(self.y_true,
                                self.y_pred,
                                average=self.average,
                                sample_weight=self.sample_weight,
                                max_fpr=self.max_fpr,
                                multi_class=self.multi_class,
                                labels=self.labels)
        return roc_auc

    def on_epoch_end(self, data: Data) -> None:
        roc_auc = self.get_auc()
        data.write_with_log(self.outputs[0], roc_auc)
