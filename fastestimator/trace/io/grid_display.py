# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import Iterable, Optional, Sequence, Union

from fastestimator.trace import Trace
from fastestimator.trace.io.batch_display import BatchDisplay
from fastestimator.trace.trace import parse_freq
from fastestimator.util.data import Data
from fastestimator.util.img_data import GridDisplay as GridDisplayF
from fastestimator.util.traceability_util import traceable


@traceable()
class GridDisplay(Trace):
    """A Trace to display of grid of images.

    Args:
        columns: A list of BatchDisplay traces to be combined into a grid. Their batch_limit, frequency, save_dir,
            mode, and ds_id arguments will be ignored in favor of the ones provided to the GridDisplay.
        batch_limit: A limit on the number of batch elements to display.
        frequency: 'batch', 'epoch', integer, or strings like '10s', '15e'. When using 'batch', writes the losses and
            metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,
            the callback will write the metrics and losses to TensorBoard every 1000 samples. You can also use strings
            like '8s' to indicate every 8 steps or '5e' to indicate every 5 epochs. You can use None to default to
            matching the log printing frequency.
        save_dir: A directory into which to save images rather than displaying them. The file names will be formatted
            as <title>_<mode>_<epoch>_<batch_idx>.html
        title: The title prefix to use if save_dir is specified.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 columns: Sequence[BatchDisplay],
                 batch_limit: Optional[int] = None,
                 frequency: Union[None, int, str] = None,
                 save_dir: Optional[str] = None,
                 title: str = "grid",
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        inputs = set()
        for column in columns:
            inputs.update(column.inputs)
        super().__init__(inputs=inputs, mode=mode, ds_id=ds_id)
        self._columns = columns
        self.frequency = None if frequency is None else parse_freq(frequency)
        self.save_dir = save_dir
        self.title = title
        self.batch_limit = batch_limit
        if self.save_dir is not None:
            self.save_dir = os.path.normpath(os.path.abspath(self.save_dir))
            os.makedirs(self.save_dir, exist_ok=True)

    def on_begin(self, data: Data) -> None:
        if self.frequency is None:
            self.frequency = parse_freq(self.system.log_steps)
        for column in self._columns:
            column.system = self.system

    def on_batch_end(self, data: Data) -> None:
        # Use the global step to match system logging behavior during train, but fall back to batch_idx for eval/test
        current_step = self.system.global_step if self.system.mode == 'train' else self.system.batch_idx
        if self.frequency.freq and self.frequency.is_step and current_step % self.frequency.freq == 0:
            columns = [col.make_image(data, batch_limit=self.batch_limit) for col in self._columns]
            display = GridDisplayF(columns=columns)
            if self.save_dir is None:
                display.show()
            else:
                filename = f'{self.title}_{self.system.mode}_{self.system.epoch_idx}_{self.system.batch_idx}.html'
                display.show(save_path=os.path.join(self.save_dir, filename))

    def on_epoch_end(self, data: Data) -> None:
        if self.frequency.freq and not self.frequency.is_step and self.system.epoch_idx % self.frequency.freq == 0:
            columns = [col.make_image(data, batch_limit=self.batch_limit) for col in self._columns]
            display = GridDisplayF(columns=columns)
            if self.save_dir is None:
                display.show()
            else:
                filename = f'{self.title}_{self.system.mode}_{self.system.epoch_idx}_{self.system.batch_idx}.html'
                display.show(save_path=os.path.join(self.save_dir, filename))
