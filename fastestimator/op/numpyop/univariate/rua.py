# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import random
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from PIL import Image, ImageOps

from fastestimator.op.numpyop.meta.one_of import OneOf
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.op.numpyop.univariate.autocontrast import AutoContrast as AutoContrastAug
from fastestimator.op.numpyop.univariate.brightness import Brightness as BrightnessAug
from fastestimator.op.numpyop.univariate.color import Color as ColorAug
from fastestimator.op.numpyop.univariate.contrast import Contrast as ContrastAug
from fastestimator.op.numpyop.univariate.posterize import Posterize as PosterizeAug
from fastestimator.op.numpyop.univariate.sharpness import Sharpness as SharpnessAug
from fastestimator.op.numpyop.univariate.shear_x import ShearX as ShearXAug
from fastestimator.op.numpyop.univariate.shear_y import ShearY as ShearYAug
from fastestimator.op.numpyop.univariate.translate_x import TranslateX as TranslateXAug
from fastestimator.op.numpyop.univariate.translate_y import TranslateY as TranslateYAug
from fastestimator.util.traceability_util import traceable


@traceable()
class Rotate(NumpyOp):
    """Rotate the input by an angle selected randomly.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        level: Factor to set the range for rotation. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.degree = level * 3.0
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_rotate(elem) for elem in data]

    def _apply_rotate(self, data: np.ndarray) -> np.ndarray:
        im = Image.fromarray(data)
        degree = random.uniform(-self.degree, self.degree)
        im = im.rotate(degree)
        return np.array(im)


@traceable()
class Identity(NumpyOp):
    """Pass the input as-is.

    Args:
        level: Placeholder argument to conform to RUA.
        inputs: Key(s) of images.
        outputs: Key(s) into which to write the images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True


@traceable()
class Equalize(NumpyOp):
    """Equalize the image histogram.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        level: Placeholder argument to conform to RUA.
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_equalize(elem) for elem in data]

    def _apply_equalize(self, data: np.ndarray) -> np.ndarray:
        im = Image.fromarray(data)
        im = ImageOps.equalize(im)
        return np.array(im)


@traceable()
class Posterize(PosterizeAug):
    """Reduce the number of bits for the image.

    Args:
        level: Factor to set the range for number of bits. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, num_bits=(round(8 - (level / 30 * 7)), 8))


@traceable()
class Solarize(NumpyOp):
    """Invert all pixel values above a threshold.

    Args:
        level: Factor to set the range for the threshold. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_limit = level / 30 * 256
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_solarize(elem) for elem in data]

    def _apply_solarize(self, data: np.ndarray) -> np.ndarray:
        threshold = 256 - round(random.uniform(0, self.loss_limit))
        data = np.where(data < threshold, data, 255 - data)
        return data


@traceable()
class AutoContrast(AutoContrastAug):
    """Adjust image contrast automatically.

    Args:
        level: Placeholder argument to conform to RUA.
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)


@traceable()
class Sharpness(SharpnessAug):
    """Randomly change the sharpness of an image.

    Args:
        level: Factor to set the range for sharpness. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, limit=level / 30 * 0.9)


@traceable()
class Contrast(ContrastAug):
    """Randomly change the contrast of an image.

    Args:
        level: Factor to set the range for contrast. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, limit=level / 30 * 0.9)


@traceable()
class Color(ColorAug):
    """Randomly change the color balance of an image.

    Args:
        level: Factor to set the range for changing color balance. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, limit=level / 30 * 0.9)


@traceable()
class Brightness(BrightnessAug):
    """Randomly change the brightness of an image.

    Args:
        level: Factor to set the range for brightness. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, limit=level / 30 * 0.9)


@traceable()
class ShearX(ShearXAug):
    """Randomly shear the image along the X axis.

    Args:
        level: Factor to set the range for shear. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, shear_coef=level / 30 * 0.5)


@traceable()
class ShearY(ShearYAug):
    """Randomly shear the image along the Y axis.

    Args:
        level: Factor to set the range for shear. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, shear_coef=level / 30 * 0.5)


@traceable()
class TranslateX(TranslateXAug):
    """Randomly shift the image along the X axis.

    Args:
        level: Factor to set the range for shift. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, shift_limit=level / 30 * 1 / 3)


@traceable()
class TranslateY(TranslateYAug):
    """Randomly shift the image along the Y axis.

    Args:
        level: Factor to set the range for shift. Must be in the range [0, 30].
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 level: Union[int, float],
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, shift_limit=level / 30 * 1 / 3)


@traceable()
class RUA(NumpyOp):
    """Apply RUA augmentation strategy.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        level: Factor to set the range for magnitude of augmentation. Must be in the range [0, 30].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 level: Union[int, float] = 18):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        aug_options = self._get_aug_list(level=level)
        self.ops = [OneOf(*aug_options) for _ in range(self._num_aug(level, len(aug_options)))]
        self.in_list, self.out_list = True, True

    def _num_aug(self, level: Union[int, float], len_aug_list: int) -> int:
        max_N = min(5, len_aug_list)
        N = min(max_N, math.ceil(level / 30 * max_N))
        return N

    def _get_aug_list(self, level: Union[int, float]) -> List[NumpyOp]:
        aug_options = [
            Rotate(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Identity(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            AutoContrast(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Equalize(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Posterize(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Solarize(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Sharpness(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Contrast(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Color(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            Brightness(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            ShearX(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            ShearY(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            TranslateX(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode),
            TranslateY(level=level, inputs=self.inputs, outputs=self.outputs, mode=self.mode)
        ]
        return aug_options

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        forward_numpyop(self.ops, data, state)
        return [data[key] for key in self.outputs]
