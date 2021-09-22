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
import os
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Union

import dill
import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class Calibrate(NumpyOp):
    """Calibrate model predictions using a given calibration function.

    This is often used in conjunction with the PBMCalibrator trace. It should be placed in the fe.Network postprocessing
    op list.

    Args:
        inputs: Key(s) of predictions to be calibrated.
        outputs: Key(s) into which to write the calibrated predictions.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        calibration_fn: The path to a dill-pickled calibration function, or an in-memory calibration function to apply.
            If a path is provided, it will be lazy-loaded and so the saved file does not need to exist already when
            training begins.
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 calibration_fn: Union[str, Callable[[np.ndarray], np.ndarray]],
                 mode: Union[None, str, Iterable[str]] = ('test', 'infer'),
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.in_list, self.out_list = True, True
        if isinstance(calibration_fn, str):
            calibration_fn = os.path.abspath(os.path.normpath(calibration_fn))
        self.calibration_fn = calibration_fn

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [
            np.squeeze(result, axis=0)
            for result in self.forward_batch([np.expand_dims(elem, axis=0) for elem in data], state)
        ]

    def forward_batch(self, data: List[Tensor], state: Dict[str, Any]) -> List[np.ndarray]:
        if isinstance(self.calibration_fn, str):
            if 'warmup' in state and state['warmup']:
                # Don't attempt to load the calibration_fn during warmup
                return data
            with open(self.calibration_fn, 'rb') as f:
                notice = f"FastEstimator-Calibrate: calibration function loaded from {self.calibration_fn}"
                self.calibration_fn = dill.load(f)
                print(notice)
        return [self.calibration_fn(to_number(elem)) for elem in data]
