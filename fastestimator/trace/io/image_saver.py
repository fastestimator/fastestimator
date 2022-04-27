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
from typing import Sequence, Union, Iterable

from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.summary.summary import Summary
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.img_data import Display, ImageDisplay
from fastestimator.util.traceability_util import traceable


@traceable()
class ImageSaver(Trace):
    """A trace that saves images to the disk.

    Args:
        inputs: Key(s) of images to be saved.
        save_dir: The directory into which to write the images.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """

    def __init__(self,
                 inputs: Union[str, Sequence[str]],
                 save_dir: str = os.getcwd(),
                 mode: Union[None, str, Iterable[str]] = ("eval", "test")) -> None:
        super().__init__(inputs=inputs, mode=mode)
        self.save_dir = save_dir

    def on_epoch_end(self, data: Data) -> None:
        self._save_images(data)

    def on_end(self, data: Data) -> None:
        self._save_images(data)

    def _save_images(self, data: Data):
        for key in self.inputs:
            if key in data:
                imgs = data[key]
                im_path = os.path.join(self.save_dir,
                                       "{}_{}_epoch_{}.png".format(key, self.system.mode, self.system.epoch_idx))
                if isinstance(imgs, Display):
                    imgs.show(save_path=im_path, verbose=False)
                    print("FastEstimator-ImageSaver: saved image to {}".format(im_path))
                elif isinstance(imgs, Summary):
                    visualize_logs([imgs], save_path=im_path, verbose=False)
                    print("FastEstimator-ImageSaver: saved image to {}".format(im_path))
                elif isinstance(imgs, (list, tuple)) and all([isinstance(img, Summary) for img in imgs]):
                    visualize_logs(imgs, save_path=im_path, verbose=False)
                    print("FastEstimator-ImageSaver: saved image to {}".format(im_path))
                else:
                    for idx, img in enumerate(imgs):
                        f = ImageDisplay(image=img, title=key)
                        im_path = os.path.join(
                            self.save_dir,
                            "{}_{}_epoch_{}_elem_{}.png".format(key, self.system.mode, self.system.epoch_idx, idx))
                        f.show(save_path=im_path, verbose=False)
                        print("FastEstimator-ImageSaver: saved image to {}".format(im_path))
