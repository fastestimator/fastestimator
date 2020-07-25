from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.backend import cast, random_mix_patch, roll
from fastestimator.op.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class CutMixBatch(TensorOp):
    """This class performs cutmix augmentation on a batch of tensors.

    In this augmentation technique patches are cut and pasted among traning images where the ground truth labels are
    also mixed proportionally to the area of the patches. This class should be used in conjunction with MixUpLoss to
    perform CutMix training, which helps to reduce over-fitting, perform object detection, and against adversarial
    attacks (https://arxiv.org/pdf/1905.04899.pdf).

    Args:
        inputs: Key(s) of the input to be cut-mixed.
        outputs: Key(s) under which to store the cut-mixed images and lambda value.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        alpha: The alpha value defining the beta distribution to be drawn from during training.
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: List[str],
                 mode: Union[None, str, Iterable[str]] = 'train',
                 alpha: Union[float, Tensor] = 1.0) -> None:
        assert alpha > 0, "Alpha value must be greater than zero"
        assert len(outputs) >= 2, "Outputs should have at least two string keys"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha
        self.beta = None
        self.uniform = None

    def build(self, framework: str) -> None:
        if framework == 'tf':
            self.alpha = tf.constant(self.alpha)
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
            self.uniform = tfp.distributions.Uniform()
        elif framework == 'torch':
            self.alpha = torch.tensor(self.alpha)
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
            self.uniform = torch.distributions.uniform.Uniform(low=0, high=1)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: Tensor, state: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        lam = self.beta.sample()
        cut_x = self.uniform.sample()
        cut_y = self.uniform.sample()
        bbox_x1, bbox_x2, bbox_y1, bbox_y2, width, height = random_mix_patch(data, cut_x, cut_y, lam=lam)
        if tf.is_tensor(data):
            patches = roll(
                data, shift=1,
                axis=0)[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :] - data[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :]
            patches = tf.pad(patches, [[0, 0], [bbox_y1, height - bbox_y2], [bbox_x1, width - bbox_x2], [0, 0]],
                             mode="CONSTANT",
                             constant_values=0)
            data = data + patches
        else:
            data[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2] = roll(data, shift=1,
                                                                axis=0)[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        # adjust lambda to match pixel ratio
        lam = 1 - cast(((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)), dtype="float32") / (width * height)
        return data, lam
