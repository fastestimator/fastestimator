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
from fastestimator.util.base_util import to_set, to_list, param_to_range


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
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        limit: Range from which the angle can be picked. If limit is a single int the range is considered from
            (0, limit).

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 limit: Union[int, Tuple[int, int]] = 30):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode, ds_id=ds_id)
        self.limit = param_to_range(limit)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """
        param_mid = (self.limit[1] + self.limit[0]) / 2
        param_extent = magnitude_coef * ((self.limit[1] - self.limit[0]) / 2)
        self.limit = (param_mid - param_extent, param_mid + param_extent)

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        degree = random.uniform(self.limit[0], self.limit[1])
        return [Rotate._apply_rotate(elem, degree) for elem in data]

    @staticmethod
    def _apply_rotate(data: np.ndarray, degree: float) -> np.ndarray:
        """Rotate the image.

        Args:
            data: The image to be modified.
            degree: Angle for image rotation.

        Returns:
            The image after applying rotation.
        """
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
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode, ds_id=ds_id)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """A method which will be invoked by the RUA Op to adjust the augmentation intensity.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """


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
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode, ds_id=ds_id)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """A method which will be invoked by the RUA Op to adjust the augmentation intensity.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [Equalize._apply_equalize(elem) for elem in data]

    @staticmethod
    def _apply_equalize(data: np.ndarray) -> np.ndarray:
        """Equalize the image histogram.

        Args:
            data: The image to be modified.

        Returns:
            The image after applying equalize.
        """
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
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
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
                 ds_id: Union[None, str, Iterable[str]] = None,
                 num_bits: Union[int,
                                 Tuple[int, int],
                                 Tuple[int, int, int],
                                 Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = 7):
        self.num_bits = num_bits
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id, num_bits=num_bits)

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """
        if isinstance(self.num_bits, tuple) and len(self.num_bits) == 3:
            num_bits = []
            for i in self.num_bits:
                num_bits.append(Posterize._range_tuple(num_bits=i, magnitude_coef=magnitude_coef))
            self.num_bits = tuple(num_bits)
        else:
            self.num_bits = Posterize._range_tuple(num_bits=self.num_bits, magnitude_coef=magnitude_coef)
        super().__init__(inputs=self.inputs,
                         outputs=self.outputs,
                         mode=self.mode,
                         ds_id=self.ds_id,
                         num_bits=self.num_bits)

    @staticmethod
    def _range_tuple(num_bits: Union[int, Tuple[int, int]], magnitude_coef: float) -> Tuple[int, int]:
        """Process num_bits for posterization based on augmentation intensity.

        Args:
            num_bits: Number of high bits.
            magnitude_coef: The desired augmentation intensity (range [0-1]).

        Returns:
            The range of high bits after adjusting augmentation intensity.
        """
        if isinstance(num_bits, tuple):
            param_mid = (num_bits[0] + num_bits[1])/2
            param_extent = magnitude_coef * ((num_bits[1] - num_bits[0])/2)
            bits_range = (round(param_mid - param_extent), round(param_mid + param_extent))
        else:
            bits_range = (round(8-(magnitude_coef*num_bits)), 8)
        return bits_range


@traceable()
class Solarize(NumpyOp):
    """Invert all pixel values above a threshold.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        threshold: Range for the solarizing threshold. If threshold is a single value 't', the range will be [0, t].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 threshold: Union[int, Tuple[int, int], float, Tuple[float, float]] = 256):
        super().__init__(inputs=to_list(inputs), outputs=to_list(outputs), mode=mode, ds_id=ds_id)
        self.threshold = threshold

    def set_rua_level(self, magnitude_coef: Union[int, float]) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
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
        """Invert all pixel values of the image above a threshold.

        Args:
            data: The image to be modified.
            threshold: Solarizing threshold.

        Returns:
            The image after applying solarize.
        """
        data = np.where(data < threshold, data, 255 - data)
        return data


@traceable()
class OneOfMultiVar(OneOf):
    """Perform one of several possible NumpyOps.

    Note that OneOfMultiVar accepts both univariate and multivariate ops and allows the list of passed NumpyOps to have
    different input and output keys. OneOfMultiVar should not be used to wrap an op whose output key(s) do not already
    exist in the data dictionary. This would result in a problem when future ops / traces attempt to reference the
    output key, but OneOfMultiVar declined to generate it. If you want to create a default value for a new key, simply
    use a LambdaOp before invoking the OneOfMultiVar.

    Args:
        *numpy_ops: A list of ops to choose between with uniform probability.
    """
    def __init__(self, *numpy_ops: NumpyOp) -> None:
        inputs = to_set(numpy_ops[0].inputs)
        outputs = to_set(numpy_ops[0].outputs)
        mode = numpy_ops[0].mode
        ds_id = numpy_ops[0].ds_id
        self.in_list = numpy_ops[0].in_list
        self.out_list = numpy_ops[0].out_list
        for op in numpy_ops[1:]:
            assert self.in_list == op.in_list, "All ops within OneOf must share the same input configuration"
            assert self.out_list == op.out_list, "All ops within OneOf must share the same output configuration"
            assert mode == op.mode, "All ops within a OneOf must share the same mode"

            for inp in op.inputs:
                inputs.add(inp)

            for out in op.outputs:
                outputs.add(out)

        # Bypassing OneOf Op's restriction of same input and output key(s) on the list of passed NumpyOps.
        super(OneOf, self).__init__(inputs=inputs.union(outputs), outputs=outputs, mode=mode, ds_id=ds_id)
        self.ops = numpy_ops

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        forward_numpyop([random.choice(self.ops)], data, state)
        return [data[key] for key in self.outputs]


@traceable()
class RUA(NumpyOp):
    """Apply RUA augmentation strategy.

    Note that all augmentation ops passed to RUA should have a set_rua_level method to modify their strength based on
    the level. Custom NumpyOps can be passed to the `choices` argument along with names of augmentations to add. Passing
    'defaults' adds the default list of augmentations along with any custom NumpyOps specified by the user.
    The default augmentations are: 'Rotate', 'Identity', 'AutoContrast', 'Equalize', 'Posterize', 'Solarize',
    'Sharpness', 'Contrast', 'Color', 'Brightness', 'ShearX', 'ShearY', 'TranslateX' and 'TranslateY'.
    To add specific augmentations from the default list, their names can be passed. Ex: 'Rotate'.
    To remove specific augmentations from the list, you can negate their names. Ex: '!Rotate' will load all the
    augmentations except 'Rotate'.

    Example combinations which are not allowed:
    choices = ['defaults', 'Rotate']        # augmentations from the default list are redundant with 'defaults'.
    choices = ['defaults', '!Rotate']       # negated augmentations automatically load the default list.
    choices = ['!Solarize', 'Rotate']       # Cannot mix negated and normal augmentations.

    RUA should not have augmentation ops whose output key(s) do not already exist in the data dictionary. This would
    result in a problem when future ops / traces attempt to reference the output key, but RUA declined to generate it.
    If you want to create a default value for a new key, simply use a LambdaOp before invoking RUA.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        choices: List of augmentations to apply.
        level: Factor to set the range for magnitude of augmentation. Must be in the range [0, 30].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 choices: Union[str, NumpyOp, List[Union[str, NumpyOp]]] = "defaults",
                 level: Union[int, float] = 18):
        self.default_aug_dict = {
            "Rotate": Rotate(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,limit=90),
            "Identity": Identity(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id),
            "AutoContrast": AutoContrast(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id),
            "Equalize": Equalize(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id),
            "Posterize": Posterize(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,num_bits=7),
            "Solarize": Solarize(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,threshold=256),
            "Sharpness": Sharpness(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,limit=0.9),
            "Contrast": Contrast(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,limit=0.9),
            "Color": Color(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,limit=0.9),
            "Brightness": Brightness(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,limit=0.9),
            "ShearX": ShearX(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,shear_coef=0.5),
            "ShearY": ShearY(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,shear_coef=0.5),
            "TranslateX": TranslateX(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,shift_limit=0.33),
            "TranslateY": TranslateY(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id,shift_limit=0.33)
        }
        aug_options = self._parse_aug_choices(magnitude_coef=(level / 30.), choices=to_list(choices))

        inputs, outputs = to_set(inputs), to_set(outputs)
        for op in aug_options:
            for inp in op.inputs:
                inputs.add(inp)

            for out in op.outputs:
                outputs.add(out)
        super().__init__(inputs=inputs.union(outputs), outputs=outputs, mode=mode, ds_id=ds_id)

        # Calculating number of augmentation to apply at each training iteration
        N_min = 1
        N_max = min(len(aug_options), 5)
        N = level * (N_max - N_min) / 30 + N_min
        N_guarantee, N_p = int(N), N % 1

        self.ops = [OneOfMultiVar(*aug_options) for _ in range(N_guarantee)]
        if N_p > 0:
            self.ops.append(Sometimes(OneOfMultiVar(*aug_options), prob=N_p))

    def _parse_aug_choices(self, magnitude_coef: float, choices: List[Union[str, NumpyOp]]) -> List[NumpyOp]:
        """Parse the augmentation choices to determine the final list of augmentations to apply.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
            choices: List of augmentations to apply.

        Returns:
            List of augmentations to apply.

        Raises:
            AssertionError: If augmentations to add and remove are mixed.
            AttributeError: If augmentation choices don't have a 'set_rua_level' method.
            ValueError: If 'defaults' is provided with augmentation strings to add or remove, or wrong names are
                provided.
        """
        custom_ops = [op for op in choices if not isinstance(op, str)]
        remove_ops = [op for op in choices if isinstance(op, str) and op.startswith("!")]
        add_ops = [op for op in choices if isinstance(op, str) and not (op.startswith("!") or (op == "defaults"))]
        aug_names = list(self.default_aug_dict.keys())

        assert len(remove_ops)==0 or len(add_ops)==0, \
            "RUA supports either add or remove ops, but not both. Found {} and {}".format(add_ops, remove_ops)

        if len(remove_ops) > 0:
            if "defaults" in choices:
                raise ValueError("Can't provide 'defaults' value with ops to remove, found: {}".format(remove_ops))
            remove_ops = [op[1:] for op in remove_ops]

            for op in remove_ops:
                if op not in aug_names:
                    raise ValueError("Unable to remove {}, list of augmentations available: {}".format(op, aug_names))

            aug_list = [aug for aug_name, aug in self.default_aug_dict.items() if aug_name not in remove_ops]
        else:
            if "defaults" in choices:
                if len(add_ops) > 0:
                    raise ValueError("Can't pass 'defaults' value with default list's ops, found: {}".format(add_ops))
                aug_list = list(self.default_aug_dict.values())
            elif len(add_ops) > 0:
                for op in add_ops:
                    if op not in aug_names:
                        raise ValueError("Unable to add {}, list of augmentations available: {}".format(op, aug_names))

                aug_list = [self.default_aug_dict[aug_name] for aug_name in add_ops]
            else:
                aug_list = []
        aug_list = aug_list + custom_ops

        for op in aug_list:
            if hasattr(op, "set_rua_level") and inspect.ismethod(getattr(op, "set_rua_level")):
                op.set_rua_level(magnitude_coef=magnitude_coef)
            else:
                raise AttributeError(
                    "RUA Augmentations should have a 'set_rua_level' method but it's not present in Op: {}".format(
                        op.__class__.__name__))

        return aug_list

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        forward_numpyop(self.ops, data, state)
        return [data[key] for key in self.outputs]
