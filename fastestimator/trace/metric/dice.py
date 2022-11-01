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
import math
from collections import defaultdict
from typing import Dict, Iterable, Optional, Union

import numpy as np

from fastestimator.backend._dice_score import dice_score
from fastestimator.backend._reduce_max import reduce_max
from fastestimator.backend._reduce_min import reduce_min
from fastestimator.summary.summary import ValWithError
from fastestimator.trace.meta._per_ds import per_ds
from fastestimator.trace.trace import Trace
from fastestimator.util.base_util import to_set
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@per_ds
@traceable()
class Dice(Trace):
    """Dice score for binary classification between y_true and y_predicted.

    Args:
        true_key: The key of the ground truth mask.
        pred_key: The key of the prediction values.
        threshold: The threshold for binarizing the prediction. Set this to 0.0 if you are using a background class. To
            skip binarization, set this to None.
        mask_overlap: Whether an individual pixel can belong to only 1 class (False) or more than 1 class
            (True). If False, a threshold of 0.0 can be used to binarize by whatever the maximum predicted class is.
        exclude_channels: A collection of channel indices to be ignored.
        channel_mapping: Optional names to give to each channel. If provided then dice will be reported per-channel
            in addition to reporting the aggregate value.
        include_std: Whether to also report standard deviations when computing dice scores.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        output_name: What to call the output from this trace (for example in the logger output).
        per_ds: Whether to automatically compute this metric individually for every ds_id it runs on, in addition to
            computing an aggregate across all ds_ids on which it runs. This is automatically False if `output_name`
            contains a "|" character.
    """

    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 threshold: Optional[float] = 0.5,
                 mask_overlap: bool = True,
                 exclude_channels: Union[None, int, Iterable[int]] = None,
                 channel_mapping: Optional[Dict[int, str]] = None,
                 include_std: bool = False,
                 mode: Union[None, str, Iterable[str]] = ("eval", "test"),
                 ds_id: Union[None, str, Iterable[str]] = None,
                 output_name: str = "Dice",
                 per_ds: bool = True) -> None:
        super().__init__(inputs=(true_key, pred_key),
                         mode=mode, outputs=output_name, ds_id=ds_id)
        self.threshold = threshold
        self.mask_overlap = mask_overlap
        self.epsilon = 1e-8
        self.per_ch_dice = {}
        self.per_ds = per_ds
        self.include_std = include_std
        self.exclude_channels = to_set(exclude_channels)
        self.channel_mapping = channel_mapping or {}

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data) -> None:
        self.per_ch_dice = defaultdict(list)

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = data[self.true_key], data[self.pred_key]

        # Do some quick input sanity checking to help prevent end user error (sparse or non-normalized masks)
        test = reduce_min(y_pred)
        assert 0 <= test, "Predicted mask values passed to the Dice trace should range from 0 to 1, but found a " \
                          f"value of {test}"
        test = reduce_max(y_pred)
        assert test <= 1, "Predicted mask values passed to the Dice trace should range from 0 to 1, but found a " \
                          f"value of {test}"
        test = reduce_min(y_true)
        assert 0 <= test, "Ground truth mask values passed to the Dice trace should range from 0 to 1, but found a " \
                          f"value of {test}"
        test = reduce_max(y_true)
        assert test <= 1, "Ground truth mask values passed to the Dice trace should range from 0 to 1, but found a " \
                          f"value of {test}"

        dice = to_number(dice_score(y_pred=y_pred,
                                    y_true=y_true,
                                    sample_average=False,
                                    channel_average=False,
                                    mask_overlap=self.mask_overlap,
                                    threshold=self.threshold,
                                    empty_nan=True,
                                    epsilon=self.epsilon))
        # Dice will be Batch x Channels
        for instance in dice:
            for idx, channel_dice in enumerate(instance):
                if math.isnan(channel_dice):
                    # If y_true and y_pred for a channel are both empty (less than 1e-4), the dice value should be
                    # excluded from the list rather than being counted as 0 in the mean (the object is missing and the
                    # model correctly identified that it is missing)
                    continue
                self.per_ch_dice[idx].append(channel_dice)

        _, n_channels = dice.shape
        dice_slices = []
        if n_channels > 1:
            for ch_idx in range(n_channels):
                if ch_idx in self.exclude_channels:
                    continue
                ch_name = ch_idx
                if ch_name in self.channel_mapping:
                    ch_name = self.channel_mapping[ch_name]
                data.write_per_instance_log(f"{self.outputs[0]}_{ch_name}", dice[:, ch_idx])
                dice_slices.append(dice[:, ch_idx])
            dice_slices = np.mean(dice_slices, axis=0)
        else:
            dice_slices = np.squeeze(dice, axis=-1)
        data.write_per_instance_log(self.outputs[0], dice_slices)

    def on_epoch_end(self, data: Data) -> None:
        means = []
        stds = []
        for ch_name, ch_vals in self.per_ch_dice.items():
            if ch_name in self.exclude_channels:
                continue
            if ch_name in self.channel_mapping:
                ch_name = self.channel_mapping[ch_name]
            mean = np.mean(ch_vals)
            means.append(mean)
            if self.include_std:
                std = np.std(ch_vals)
                stds.append(std)
            # If there are multiple channels and the user has provided channel names, then report each channel
            if len(self.per_ch_dice.items()) > 1 and self.channel_mapping:
                if self.include_std:
                    data.write_with_log(f"{self.outputs[0]}_{ch_name}", ValWithError(mean - std, mean, mean + std))
                else:
                    data.write_with_log(f"{self.outputs[0]}_{ch_name}", mean)
        mean = np.mean(means)
        if self.include_std:
            std = np.mean(stds)
            data.write_with_log(self.outputs[0], ValWithError(mean - std, mean, mean + std))
        else:
            data.write_with_log(self.outputs[0], mean)
        self.per_ch_dice.clear()
