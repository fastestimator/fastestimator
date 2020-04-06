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

from typing import Set, Union, Sequence

import matplotlib.pyplot as plt

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.xai.util import show_image, XaiData


class ImageViewer(Trace):
    def __init__(self, inputs: Union[str, Sequence[str]], mode: Union[str, Set[str]] = ("eval", "test")) -> None:
        super().__init__(inputs=inputs, mode=mode)

    def on_epoch_end(self, data: Data):
        for key in self.inputs:
            if key in data:
                imgs = data[key]
                if isinstance(imgs, XaiData):
                    imgs.paint_figure()
                    plt.show()
                else:
                    for idx, img in enumerate(imgs):
                        show_image(img, title="{}_{}".format(key, idx))
                        plt.show()
