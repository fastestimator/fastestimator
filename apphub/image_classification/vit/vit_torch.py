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
"""
Vision Transformer PyTorch Implementation
"""
import tempfile

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import fastestimator as fe
from fastestimator.dataset.data import cifair10, cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class ViTEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_channels=3, em_dim=768, drop=0.1) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image size must be an integer multiple of patch size"
        self.patch_embedding = nn.Conv2d(num_channels, em_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, em_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, em_dim))
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # [B,C, H, W] -> [B, num_patches, em_dim]
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)  # [B, num_patches+1, em_dim]
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, num_layers, image_size, patch_size, num_channels, em_dim, drop, num_heads, ff_dim):
        super().__init__()
        self.embedding = ViTEmbeddings(image_size, patch_size, num_channels, em_dim, drop)
        encoder_layer = TransformerEncoderLayer(em_dim,
                                                nhead=num_heads,
                                                dim_feedforward=ff_dim,
                                                activation='gelu',
                                                dropout=drop)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(em_dim, eps=1e-6)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Switch batch and sequence length dimension for pytorch convention
        x = self.encoder(x)
        x = self.layernorm(x[0])
        return x


class ViTModel(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers=12,
                 image_size=224,
                 patch_size=16,
                 num_channels=3,
                 em_dim=768,
                 drop=0.1,
                 num_heads=12,
                 ff_dim=3072):
        super().__init__()
        self.vit_encoder = ViTEncoder(num_layers=num_layers,
                                      image_size=image_size,
                                      patch_size=patch_size,
                                      num_channels=num_channels,
                                      em_dim=em_dim,
                                      drop=drop,
                                      num_heads=num_heads,
                                      ff_dim=ff_dim)
        self.linear_classifier = nn.Linear(em_dim, num_classes)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.linear_classifier(x)
        return x


def pretrain(batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             max_train_steps_per_epoch=None,
             max_eval_steps_per_epoch=None):
    train_data, eval_data = cifair100.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])
    model = fe.build(
        model_fn=lambda: ViTModel(num_classes=100,
                                  image_size=32,
                                  patch_size=4,
                                  num_layers=6,
                                  num_channels=3,
                                  em_dim=256,
                                  num_heads=8,
                                  ff_dim=512),
        optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.9, weight_decay=1e-4))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    estimator.fit(warmup=False)
    return model


def finetune(pretrained_model,
             batch_size,
             epochs,
             model_dir=tempfile.mkdtemp(),
             max_train_steps_per_epoch=None,
             max_eval_steps_per_epoch=None):
    train_data, eval_data = cifair10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])
    model = fe.build(
        model_fn=lambda: ViTModel(num_classes=100,
                                  image_size=32,
                                  patch_size=4,
                                  num_layers=6,
                                  num_channels=3,
                                  em_dim=256,
                                  num_heads=8,
                                  ff_dim=512),
        optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.9, weight_decay=1e-4))
    # load the encoder's weight
    model.vit_encoder.load_state_dict(pretrained_model.vit_encoder.state_dict())
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    estimator.fit(warmup=False)


def fastestimator_run(batch_size=128,
                      pretrain_epochs=100,
                      finetune_epochs=1,
                      max_train_steps_per_epoch=None,
                      max_eval_steps_per_epoch=None):
    pretrained_model = pretrain(batch_size=batch_size,
                                epochs=pretrain_epochs,
                                max_train_steps_per_epoch=max_train_steps_per_epoch,
                                max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    finetune(pretrained_model,
             batch_size=batch_size,
             epochs=finetune_epochs,
             max_train_steps_per_epoch=max_train_steps_per_epoch,
             max_eval_steps_per_epoch=max_eval_steps_per_epoch)


if __name__ == "__main__":
    fastestimator_run()
