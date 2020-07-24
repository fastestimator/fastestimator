from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.backend import random_mix_patch, roll
from fastestimator.op.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class CutMixBatch(TensorOp):
    """This class perform cutmix augmentation on batch of tensors.

    In this augmentation technique patches are cut and pasted among traning images where the ground truth labels are
    also mixed proportionally to the area of the patches. This class should be used in conjunction with MixUpLoss to
    perform CutMix training, which helps to reduce over-fitting, perform object detection, and against adversarial
    attacks (https://arxiv.org/pdf/1905.04899.pdf)

    Args:
        inputs: key of the input to be cut-mixed
        outputs: key to store the cut-mixed input
        mode: what mode to execute in. Probably 'train'
        alpha: the alpha value defining the beta distribution to be drawn from during training
        framework: which framework the current Op will be executing in. Either 'tf' or 'torch'
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 alpha: Union[float, Tensor] = 1.0,
                 framework: str = 'tf') -> None:
        assert alpha > 0, "Alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha
        self.beta = None
        self.uniform = None
        self.build(framework)

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
        """ Forward method to perform cutmix batch augmentation
        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.
        Returns:
            Tuple of Cut-Mixed batch data and lambda
        """
        lam = self.beta.sample()
        uniform_sample = self.uniform.sample()
        bbox_x1, bbox_x2, bbox_y1, bbox_y2, width, height = random_mix_patch(data, lam, uniform_sample)
        if tf.is_tensor(data):
            patches = roll(
                data, shift=1,
                axis=0)[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :] - data[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :]
            patches = tf.pad(patches, [[0, 0], [bbox_y1, height - bbox_y2], [bbox_x1, width - bbox_x2], [0, 0]],
                             mode="CONSTANT",
                             constant_values=0)
            data = data + patches
            # adjust lambda to match pixel ratio
            lam = tf.dtypes.cast(1.0 - (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) / (width * height), tf.float32)
        else:
            data[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2] = roll(data, shift=1,
                                                                axis=0)[:, :, bbox_y1:bbox_y2, bbox_x1:bbox_x2]
            lam = 1 - ((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)).type(torch.float32) / (width * height)
        return data, lam
