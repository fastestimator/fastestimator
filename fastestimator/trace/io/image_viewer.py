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
from typing import Sequence, Set, Union

import matplotlib.pyplot as plt

from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.summary.summary import Summary
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.img_data import ImgData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import show_image


@traceable()
class ImageViewer(Trace):
    """A trace that interrupts your training in order to display images on the screen.

    This class is useful primarily for Jupyter Notebook, or for debugging purposes.

    Args:
        inputs: Key(s) of images to be displayed.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        width: The width in inches of the figure.
        height: The height in inches of the figure.
    """
    def __init__(self,
                 inputs: Union[str, Sequence[str]],
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 width: int = 12,
                 height: int = 6) -> None:
        super().__init__(inputs=inputs, mode=mode)
        plt.rcParams['figure.figsize'] = [width, height]

    def on_epoch_end(self, data: Data) -> None:
        self._display_images(data)

    def on_end(self, data: Data) -> None:
        self._display_images(data)

    def _display_images(self, data: Data) -> None:
        """A method to render images to the screen.

        Args:
            data: Data possibly containing images to render.
        """
        for key in self.inputs:
            if key in data:
                imgs = data[key]
                if isinstance(imgs, ImgData):
                    fig = imgs.paint_numpy(dpi=96)
                    plt.imshow(fig[0])
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                elif isinstance(imgs, Summary):
                    visualize_logs([imgs])
                elif isinstance(imgs, (list, tuple)) and all([isinstance(img, Summary) for img in imgs]):
                    visualize_logs(imgs)
                else:
                    for idx, img in enumerate(imgs):
                        show_image(img, title="{}_{}".format(key, idx))
                        plt.show()
