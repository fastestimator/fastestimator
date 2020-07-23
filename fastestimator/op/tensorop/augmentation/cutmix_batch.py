from typing import Any, Dict, Iterable, List, TypeVar, Union, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.op.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class CutMixBatch(TensorOp):
    """ This class should be used in conjunction with MixUpLoss to perform CutMix training, which helps to reduce
    over-fitting, perform object detection, and against adversarial attacks (https://arxiv.org/pdf/1905.04899.pdf)
    Args:
        inputs: key of the input to be cut-mixed
        outputs: key to store the cut-mixed input
        mode: what mode to execute in. Probably 'train'
        alpha: the alpha value defining the beta distribution to be drawn from during training
    """
    def __init__(self, inputs=None, outputs=None, mode=None, alpha=1.0, framework='tf'):
        assert alpha > 0, "Mixup alpha value must be greater than zero"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if framework == 'tf':
            self.alpha = tf.constant(alpha)
            self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
            self.uniform = tfp.distributions.Uniform()
        elif framework == 'torch':
            self.alpha = torch.tensor(alpha)
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
            self.uniform = torch.distributions.uniform.Uniform(low=0, high=1)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def forward(self, data: Tensor, state: Dict[str, Any]) -> Tuple(Tensor, Tensor):
        """ Forward method to perform cutmix batch augmentation
        Args:
            data: Batch data to be augmented
            state: Information about the current execution context.
        Returns:
            Cut-Mixed batch data
        """
        lam = self.beta.sample()
        if tf.is_tensor(data):
            _, height, width, _ = data.shape
            rx = width * self.uniform.sample()
            ry = height * self.uniform.sample()
            rw = width * tf.sqrt(1 - lam)
            rh = height * tf.sqrt(1 - lam)
            x1 = tf.dtypes.cast(tf.round(tf.math.maximum(rx - rw / 2, 0)), tf.int32)
            x2 = tf.dtypes.cast(tf.round(tf.math.minimum(rx + rw / 2, width)), tf.int32)
            y1 = tf.dtypes.cast(tf.round(tf.math.maximum(ry - rh / 2, 0)), tf.int32)
            y2 = tf.dtypes.cast(tf.round(tf.math.minimum(ry + rh / 2, height)), tf.int32)

            patches = tf.roll(data, shift=1, axis=0)[:, y1:y2, x1:x2, :] - data[:, y1:y2, x1:x2, :]
            patches = tf.pad(patches, [[0, 0], [y1, height - y2], [x1, width - x2], [0, 0]],
                             mode="CONSTANT",
                             constant_values=0)
            return data + patches, lam
        else:
            _, _, height, width = data.shape
            rx = width * self.uniform.sample()
            ry = height * self.uniform.sample()
            rw = width * torch.sqrt(1 - lam)
            rh = height * torch.sqrt(1 - lam)
            x1 = torch.round(torch.clamp(rx - rw / 2, min=0)).type(torch.int32)
            x2 = torch.round(torch.clamp(rx + rw / 2, max=width)).type(torch.int32)
            y1 = torch.round(torch.clamp(ry - rh / 2, min=0)).type(torch.int32)
            y2 = torch.round(torch.clamp(ry + rh / 2, max=height)).type(torch.int32)

            data[:, :, y1:y2, x1:x2] = torch.roll(data, shifts=1, dims=0)[:, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1).type(torch.float32) / (data.size()[-1] * data.size()[-2]))
            lam = torch.tensor(lam)
            return data, lam
