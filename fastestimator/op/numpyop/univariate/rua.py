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
import inspect
import random
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from fastestimator.op.numpyop.meta.one_of import OneOf
from fastestimator.op.numpyop.meta.sometimes import Sometimes
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.op.numpyop.univariate.autocontrast import AutoContrast
from fastestimator.op.numpyop.univariate.brightness import Brightness
from fastestimator.op.numpyop.univariate.color import Color
from fastestimator.op.numpyop.univariate.contrast import Contrast
from fastestimator.op.numpyop.univariate.posterize import Posterize as PosterizeAug
from fastestimator.op.numpyop.univariate.sharpness import Sharpness
from fastestimator.op.numpyop.univariate.shear_x import ShearX
from fastestimator.op.numpyop.univariate.shear_y import ShearY
from fastestimator.op.numpyop.univariate.translate_x import TranslateX
from fastestimator.op.numpyop.univariate.translate_y import TranslateY
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import param_to_range, to_list


@traceable()
class Rotate(NumpyOp):
    """Rotate the input by an angle selected randomly.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        limit: Range from which the angle can be picked. If limit is a single int the range is considered from
            (0, limit).

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 limit: Union[int, Tuple[int, int]] = 30):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode)
        self.limit = param_to_range(limit)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intentity based on the magnitude_coef.

        This method is specifically designed to be invoked by RUA Op.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
        """
        param_mid = (self.limit[1] + self.limit[0]) / 2
        param_extent = magnitude_coef * ((self.limit[1] - self.limit[0]) / 2)
        self.limit = (param_mid - param_extent, param_mid + param_extent)

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        degree = random.uniform(self.limit[0], self.limit[1])
        return [Rotate._apply_rotate(elem, degree) for elem in data]

    @staticmethod
    def _apply_rotate(data: np.ndarray, degree: float) -> np.ndarray:
        im = Image.fromarray(data)
        im = im.rotate(degree)
        return np.array(im)


@traceable()
class Identity(NumpyOp):
    """Pass the input as-is.

    Args:
        inputs: Key(s) of images.
        outputs: Key(s) into which to write the images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """A method which will be invoked by RUA Op to adjust the augmentation intensity.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
        """
        pass


@traceable()
class Equalize(NumpyOp):
    """Equalize the image histogram.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """A method which will be invoked by RUA Op to adjust the augmentation intensity.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
        """
        pass

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [Equalize._apply_equalize(elem) for elem in data]

    @staticmethod
    def _apply_equalize(data: np.ndarray) -> np.ndarray:
        im = Image.fromarray(data)
        im = ImageOps.equalize(im)
        return np.array(im)


@traceable()
class Posterize(PosterizeAug):
    """Reduce the number of bits for the image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        num_bits: Number of high bits. If num_bits is a single value, the range will be [num_bits, num_bits]. A triplet
            of ints will be interpreted as [r, g, b], and a triplet of pairs as [[r1, r1], [g1, g2], [b1, b2]]. Must be
            in the range [0, 8].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 num_bits: Union[int,
                                 Tuple[int, int],
                                 Tuple[int, int, int],
                                 Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = 7):
        self.num_bits = num_bits
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, num_bits=num_bits)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intentity based on the magnitude_coef.

        This method is specifically designed to be invoked by RUA Op.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
        """
        super().__init__(inputs=self.inputs,
                         outputs=self.outputs,
                         mode=self.mode,
                         num_bits=round(8 - (magnitude_coef * self.num_bits)))


@traceable()
class Solarize(NumpyOp):
    """Invert all pixel values above a threshold.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        threshold: Range for the solarizing threshold. If threshold is a single value 't', the range will be [0, t].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 threshold: Union[int, Tuple[int, int], float, Tuple[float, float]] = 256):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode)
        self.threshold = threshold

    def set_rua_level(self, magnitude_coef: Union[int, float]) -> None:
        """Set the augmentation intentity based on the magnitude_coef.

        This method is specifically designed to be invoked by RUA Op.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
        """
        if isinstance(self.threshold, tuple):
            self.threshold = magnitude_coef * (self.threshold[1] - self.threshold[0]) + self.threshold[0]
        else:
            self.threshold = magnitude_coef * self.threshold

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        if isinstance(self.threshold, tuple):
            threshold = 256 - round(random.uniform(self.threshold[0], self.threshold[1]))
        else:
            threshold = 256 - round(random.uniform(0, self.threshold))
        return [Solarize._apply_solarize(elem, threshold) for elem in data]

    @staticmethod
    def _apply_solarize(data: np.ndarray, threshold: int) -> np.ndarray:
        data = np.where(data < threshold, data, 255 - data)
        return data


@traceable()
class RUA(NumpyOp):
    """Apply RUA augmentation strategy.

    Note that all augmentations ops passed to RUA should have a set_rua_level method to modify the magnitude of
    augmentations based on the level. Custom NumpyOp can be passed to the choices argument along with names of
    augmentations to add. Passing 'defaults' adds the default list of augmentations along with any custom NumpyOps
    specified by the user.
    List of default augmentations available are: 'Rotate', 'Identity', 'AutoContrast', 'Equalize', 'Posterize',
    'Solarize', 'Sharpness', 'Contrast', 'Color', 'Brightness', 'ShearX', 'ShearY', 'TranslateX' and 'TranslateY'.
    To add specific augmentations from the default list, their names can be passed. Ex: 'Rotate'.
    To remove specific augmentations from the list, negate of the augmentation name can be provided. Ex: '!Rotate' will
    load all the augmentations except 'Rotate'.

    Few cases which are not allowed:
    choices = ['defaults', 'Rotate']        # augmentations from the default list are not allowed along with 'defaults'
    choices = ['defaults', '!Rotate']       # negated augmentations automatically load the default list.
    choices = ['!Solarize', 'Rotate']       # Cannot mix negated and normal augmentations

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        choices: list of augmentations to apply.
        level: Factor to set the range for magnitude of augmentation. Must be in the range [0, 30].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 choices: Union[str, NumpyOp, List[Union[str, NumpyOp]]] = "defaults",
                 level: Union[int, float] = 18):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode)
        self.default_aug_list = [
            Rotate(inputs=inputs, outputs=outputs, mode=mode, limit=30),
            Identity(inputs=inputs, outputs=outputs, mode=mode),
            AutoContrast(inputs=inputs, outputs=outputs, mode=mode),
            Equalize(inputs=inputs, outputs=outputs, mode=mode),
            Posterize(inputs=inputs, outputs=outputs, mode=mode, num_bits=7),
            Solarize(inputs=inputs, outputs=outputs, mode=mode, threshold=256),
            Sharpness(inputs=inputs, outputs=outputs, mode=mode, limit=0.9),
            Contrast(inputs=inputs, outputs=outputs, mode=mode, limit=0.9),
            Color(inputs=inputs, outputs=outputs, mode=mode, limit=0.9),
            Brightness(inputs=inputs, outputs=outputs, mode=mode, limit=0.9),
            ShearX(inputs=inputs, outputs=outputs, mode=mode, shear_coef=0.5),
            ShearY(inputs=inputs, outputs=outputs, mode=mode, shear_coef=0.5),
            TranslateX(inputs=inputs, outputs=outputs, mode=mode, shift_limit=0.33),
            TranslateY(inputs=inputs, outputs=outputs, mode=mode, shift_limit=0.33)
        ]
        aug_options = self._parse_aug_choices(magnitude_coef=(level / 30.), choices=to_list(choices))

        # Calculating number of augmentation to apply at each training iteration
        N_min = 1
        N_max = min(len(aug_options), 5)
        N = level * (N_max - N_min) / 30 + N_min
        N_guarantee, N_p = int(N), N % 1

        self.ops = [OneOf(*aug_options) for _ in range(N_guarantee)]
        if N_p > 0:
            self.ops.append(Sometimes(OneOf(*aug_options), prob=N_p))

    def _parse_aug_choices(self, magnitude_coef: float, choices: List[Union[str, NumpyOp]]) -> List[NumpyOp]:
        """Parse the augmentation choices to determine the final list of augmentations to apply.

        Args:
            magnitude_coef: Factor to set the range for magnitude of augmentation.
            choices: list of augmentations to apply.

        Returns:
            List of augmentations to apply.

        Raises:
            AssertionError: If augmentations to add and remove are mixed.
            AttributeError: If augmentation choices don't have a 'set_rua_level' method.
            ValueError: If 'defaults' is provided with augmentation strings to add or remove, or wrong names provided.
        """
        custom_ops = [op for op in choices if not isinstance(op, str)]
        remove_ops = [op for op in choices if isinstance(op, str) and op.startswith("!")]
        add_ops = [op for op in choices if isinstance(op, str) and not (op.startswith("!") or (op == "defaults"))]

        assert len(remove_ops)==0 or len(add_ops)==0, \
            "Either add or remove ops, not both, found {} and {}".format(add_ops, remove_ops)

        if len(remove_ops) > 0:
            if "defaults" in choices:
                raise ValueError("Can't provide 'defaults' value with ops to remove, found: {}".format(remove_ops))
            aug_list = self.default_aug_list
            aug_names = [op.__class__.__name__ for op in aug_list]
            remove_ops = [op[1:] for op in remove_ops]

            for op in remove_ops:
                if op not in aug_names:
                    raise ValueError("Unable to remove {}, list of augmentations available: {}".format(op, aug_names))

            aug_list = [aug for aug in aug_list if aug.__class__.__name__ not in remove_ops]
        else:
            if "defaults" in choices:
                if len(add_ops) > 0:
                    raise ValueError("Can't pass 'defaults' value with default list's ops, found: {}".format(add_ops))
                aug_list = self.default_aug_list
            elif len(add_ops) > 0:
                aug_list = self.default_aug_list
                aug_names = [op.__class__.__name__ for op in aug_list]

                for op in add_ops:
                    if op not in aug_names:
                        raise ValueError("Unable to add {}, list of augmentations available: {}".format(op, aug_names))

                aug_list = [aug for aug in aug_list if aug.__class__.__name__ in add_ops]
            else:
                aug_list = []
        aug_list = aug_list + custom_ops

        for op in aug_list:
            if hasattr(op, "set_rua_level") and inspect.ismethod(getattr(op, "set_rua_level")):
                op.set_rua_level(magnitude_coef=magnitude_coef)
            else:
                raise AttributeError(
                    "RUA Augmentations should have 'set_rua_level' method but it's not present in Op: {}".format(
                        op.__class__.__name__))

        return aug_list

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        forward_numpyop(self.ops, data, state)
        return [data[key] for key in self.outputs]
