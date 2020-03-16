import math
import tempfile
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data.mnist import load_data
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import Binarize, ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver

LATENT_DIM = 50


class SplitOp(TensorOp):
    """To split the infer net output into two """
    def forward(self, data: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.split(data, LATENT_DIM, dim=1)
        return mean, logvar


class ReparameterizepOp(TensorOp):
    """Reparameterization trick. Ensures grads pass thru the sample to the infer net parameters"""
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        mean, logvar = data
        eps = torch.randn(mean.shape, device=mean.device)
        A = torch.exp(logvar * 0.5)
        output = eps * A + mean
        return output


class CVAELoss(TensorOp):
    """Convolutional variational auto-endcoder loss"""
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        cross_ent, mean, logvar, z = data
        cross_ent = cross_ent * (1 * 28 * 28)
        logpz = self._log_normal_pdf(z, 0, 0)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        total_loss = cross_ent + torch.mean(-logpz + logqz_x, dim=0)
        return total_loss

    @staticmethod
    def _log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = math.log(2 * math.pi)
        if isinstance(logvar, torch.Tensor):
            exp_logvar = torch.exp(-logvar)
        else:
            exp_logvar = math.exp(-logvar)
        return torch.sum(-.5 * ((sample - mean)**2. * exp_logvar + logvar + log2pi), dim=raxis)


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.fc1 = nn.Linear(2304, LATENT_DIM * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.relu(x)
        x = self.conv2(x)
        x = fn.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, 7 * 7 * 32)
        self.conv1 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = fn.relu(x)
        x = x.view(-1, 32, 7, 7)
        x = self.conv1(x)
        x = fn.relu(x)
        x = self.conv2(x)
        x = fn.relu(x)
        x = self.conv3(x)

        return x


def get_estimator(batch_size=100, epochs=100, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    train_data, test_data = load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=test_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x", axis=0),
            Minmax(inputs="x", outputs="x"),
            Binarize(inputs="x", outputs="x", threshold=0.5),
        ])

    encode_model = fe.build(model_fn=EncoderNet, optimizer_fn="adam", model_names="encoder")
    decode_model = fe.build(model_fn=DecoderNet, optimizer_fn="adam", model_names="decoder")

    network = fe.Network(ops=[
        ModelOp(model=encode_model, inputs="x", outputs="meanlogvar"),
        SplitOp(inputs="meanlogvar", outputs=("mean", "logvar")),
        ReparameterizepOp(inputs=("mean", "logvar"), outputs="z"),
        ModelOp(model=decode_model, inputs="z", outputs="x_logit"),
        CrossEntropy(inputs=("x_logit", "x"), outputs="cross_entropy", from_logits=True),
        CVAELoss(inputs=("cross_entropy", "mean", "logvar", "z"), outputs="loss"),
        UpdateOp(model=encode_model, loss_name="loss"),
        UpdateOp(model=decode_model, loss_name="loss"),
    ])

    traces = [
        BestModelSaver(model=encode_model, save_dir=save_dir), BestModelSaver(model=decode_model, save_dir=save_dir)
    ]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
