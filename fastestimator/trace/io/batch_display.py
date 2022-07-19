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
from typing import Iterable, Optional, Union

from fastestimator.trace import Trace
from fastestimator.trace.trace import parse_freq
from fastestimator.util.data import Data
from fastestimator.util.img_data import BatchDisplay as BatchDisplayF


class BatchDisplay(Trace):
    """A Trace to display a batch of images.

    Args:
        image: Key corresponding to a batch of images to be displayed.
        text: Key corresponding to text to be printed in the center of the figure.
        masks: Key corresponding to masks to be displayed over an image.
        bboxes: Key corresponding to bounding boxes to be displayed over the image.
        keypoints: Key corresponding to keypoints to be displayed over the image.
        color_map: How to color 1-channel images. Options from: https://plotly.com/python/builtin-colorscales/
        title: The title of the generated figure. If None it defaults to any image/text/mask/bbox/keypoint key which was
            provided (in that order).
        batch_limit: A limit on the number of batch elements to display.
        frequency: 'batch', 'epoch', integer, or strings like '10s', '15e'. When using 'batch', writes the losses and
            metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,
            the callback will write the metrics and losses to TensorBoard every 1000 samples. You can also use strings
            like '8s' to indicate every 8 steps or '5e' to indicate every 5 epochs. You can use None to default to
            matching the log printing frequency.
        save_dir: A directory into which to save images rather than displaying them. The file names will be formatted
            as <title>_<mode>_<epoch>_<batch_idx>.html
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 image: Optional[str] = None,
                 text: Optional[str] = None,
                 masks: Optional[str] = None,
                 bboxes: Optional[str] = None,
                 keypoints: Optional[str] = None,
                 color_map: str = "greys",
                 title: Optional[str] = None,
                 batch_limit: Optional[int] = None,
                 frequency: Union[None, int, str] = None,
                 save_dir: Optional[str] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        inputs = {image, text, masks, bboxes, keypoints}
        inputs.discard(None)
        assert inputs, "At least one input key must be specified"
        super().__init__(inputs=inputs, mode=mode, ds_id=ds_id)
        self._image = image
        self._text = text
        self._masks = masks
        self._bboxes = bboxes
        self._keypoints = keypoints
        self._title = title or image or text or masks or bboxes or keypoints or None
        self._color_map = color_map
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = os.path.normpath(os.path.abspath(self.save_dir))
            os.makedirs(self.save_dir, exist_ok=True)
        self.frequency = None if frequency is None else parse_freq(frequency)
        self.batch_limit = batch_limit

    def on_begin(self, data: Data) -> None:
        if self.frequency is None:
            self.frequency = parse_freq(self.system.log_steps)

    def make_image(self, data: Data, batch_limit: Optional[int] = None) -> BatchDisplayF:
        display = BatchDisplayF(image=data[self._image] if self._image else None,
                                text=data[self._text] if self._text else None,
                                masks=data[self._masks] if self._masks else None,
                                bboxes=data[self._bboxes] if self._bboxes else None,
                                keypoints=data[self._keypoints] if self._keypoints else None,
                                color_map=self._color_map,
                                title=self._title)
        if batch_limit and batch_limit < display.batch_size:
            display.batch = display.batch[:batch_limit]
            display.batch_size = batch_limit
        return display

    def on_batch_end(self, data: Data) -> None:
        # Use the global step to match system logging behavior during train, but fall back to batch_idx for eval/test
        current_step = self.system.global_step if self.system.mode == 'train' else self.system.batch_idx
        if self.frequency.freq and self.frequency.is_step and current_step % self.frequency.freq == 0:
            display = self.make_image(data, batch_limit=self.batch_limit)
            if self.save_dir is None:
                display.show()
            else:
                title = self._title or self._image
                filename = f'{title}_{self.system.mode}_{self.system.epoch_idx}_{self.system.batch_idx}.html'
                display.show(save_path=os.path.join(self.save_dir, filename))

    def on_epoch_end(self, data: Data) -> None:
        if self.frequency.freq and not self.frequency.is_step and self.system.epoch_idx % self.frequency.freq == 0:
            display = self.make_image(data, batch_limit=self.batch_limit)
            if self.save_dir is None:
                display.show()
            else:
                title = self._title or self._image
                filename = f'{title}_{self.system.mode}_{self.system.epoch_idx}_{self.system.batch_idx}.html'
                display.show(save_path=os.path.join(self.save_dir, filename))
