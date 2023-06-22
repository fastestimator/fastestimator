# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
The FastEstimator implementation of LeVIT model ref: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/levit.py,
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Note that we use the ciFAIR10 dataset instead (https://cvjena.github.io/cifair/)
"""
import itertools
import tempfile

import numpy as np
import torch
from torch.nn.init import trunc_normal_

import fastestimator as fe
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

specification = {
    'LeViT_128S': {
        'embed_dim': [128, 256, 384],
        'key_dim': 16,
        'num_heads': [4, 6, 8],
        'depth': [2, 3, 4],
        'drop_path': 0,
        'weights': 'https://huggingface.co/facebook/levit-128S/resolve/main/pytorch_model.bin'
    },
    'LeViT_256': {
        'embed_dim': [256, 384, 512],
        'key_dim': 32,
        'num_heads': [4, 6, 8],
        'depth': [4, 4, 4],
        'drop_path': 0,
        'weights': 'https://huggingface.co/facebook/levit-256/resolve/main/pytorch_model.bin'
    },
    'LeViT_384': {
        'embed_dim': [384, 512, 768],
        'key_dim': 32,
        'num_heads': [6, 9, 12],
        'depth': [4, 4, 4],
        'drop_path': 0.1,
        'weights': 'https://huggingface.co/facebook/levit-384/resolve/main/pytorch_model.bin'
    },
}


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1):
        super().__init__()
        self.convolution = torch.nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size,
                                           stride,
                                           padding,
                                           dilation,
                                           groups,
                                           bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

        torch.nn.init.constant_(self.batch_norm.weight, bn_weight_init)

    def forward(self, x):
        return self.batch_norm(self.convolution(x))


class Backbone(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.convolution_layer1 = ConvNorm(in_channels,
                                           out_channels // 8,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)
        self.activation_layer1 = torch.nn.Hardswish()
        self.convolution_layer2 = ConvNorm(out_channels // 8,
                                           out_channels // 4,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)
        self.activation_layer2 = torch.nn.Hardswish()
        self.convolution_layer3 = ConvNorm(out_channels // 4,
                                           out_channels // 2,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)
        self.activation_layer3 = torch.nn.Hardswish()
        self.convolution_layer4 = ConvNorm(out_channels // 2,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)

    def forward(self, x):
        x = self.activation_layer1(self.convolution_layer1(x))
        x = self.activation_layer2(self.convolution_layer2(x))
        x = self.activation_layer3(self.convolution_layer3(x))
        return self.convolution_layer4(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bn_weight_init=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        torch.nn.init.constant_(self.batch_norm.weight, bn_weight_init)

    def forward(self, x):
        x = self.linear(x)
        return self.batch_norm(x.flatten(0, 1)).reshape_as(x)


class Downsample(torch.nn.Module):
    def __init__(self, stride, resolution, use_pool=False):
        super().__init__()
        self.stride = stride
        self.resolution = resolution
        self.pool = torch.nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False) if use_pool else None

    def forward(self, x):
        batch_size, _, channels = x.shape
        x = x.view(batch_size, self.resolution, self.resolution, channels)
        if self.pool is not None:
            x = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            x = x[:, ::self.stride, ::self.stride]

        return x.reshape(batch_size, -1, channels)


class Residual(torch.nn.Module):
    def __init__(self, module, drop_rate):
        super().__init__()
        self.module = module
        self.drop_out = torch.nn.Dropout(p=drop_rate)

    def forward(self, x):
        if self.training:
            return x + self.drop_out(self.module(x))
        else:
            return x + self.module(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_attention_heads=8, attention_ratio=4, resolution=14):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio

        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads

        self.queries_keys_values = LinearNorm(dim, self.out_dim_keys_values)
        self.activation = torch.nn.Hardswish()
        self.projection = LinearNorm(self.out_dim_projection, dim, bn_weight_init=0)
        points = list(itertools.product(range(resolution), range(resolution)))

        len_points = len(points)
        attention_offsets, indices = {}, []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        self.attention_bias_cache = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(indices).view(len_points, len_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, hidden_state):
        batch_size, seq_length, _ = hidden_state.shape
        queries_keys_values = self.queries_keys_values(hidden_state)
        query, key, value = queries_keys_values.view(
            batch_size, seq_length, self.num_attention_heads, -1).split([
                self.key_dim, self.key_dim, self.attention_ratio * self.key_dim
            ],
                                                                        dim=3)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        attention = attention.softmax(dim=-1)
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, seq_length, self.out_dim_projection)
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state


class AttentionDownsample(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        key_dim,
        num_attention_heads,
        attention_ratio,
        stride,
        resolution_in,
        resolution_out,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.resolution_out = resolution_out
        # resolution_in is the intial resolution, resoloution_out is final resolution after downsampling
        self.keys_values = LinearNorm(input_dim, self.out_dim_keys_values)
        self.queries_subsample = Downsample(stride, resolution_in)
        self.queries = LinearNorm(input_dim, key_dim * num_attention_heads)
        self.activation = torch.nn.Hardswish()
        self.projection = LinearNorm(self.out_dim_projection, output_dim)

        self.attention_bias_cache = {}

        points = list(itertools.product(range(resolution_in), range(resolution_in)))
        points_ = list(itertools.product(range(resolution_out), range(resolution_out)))
        len_points, len_points_ = len(points), len(points_)
        attention_offsets, indices = {}, []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(indices).view(len_points_, len_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, hidden_state):
        batch_size, seq_length, _ = hidden_state.shape
        key, value = (self.keys_values(hidden_state).view(
            batch_size, seq_length, self.num_attention_heads,
            -1).split([self.key_dim, self.attention_ratio * self.key_dim],
                      dim=3))
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        query = self.queries(self.queries_subsample(hidden_state))
        query = query.view(batch_size, self.resolution_out**2, self.num_attention_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        attention = attention.softmax(dim=-1)
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, -1, self.out_dim_projection)
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state


class MLP(torch.nn.Module):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_up = LinearNorm(input_dim, hidden_dim)
        self.activation = torch.nn.Hardswish()
        self.linear_down = LinearNorm(hidden_dim, input_dim)

    def forward(self, hidden_state):
        hidden_state = self.linear_up(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.linear_down(hidden_state)
        return hidden_state


class NormLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, std=0.02, drop=0.):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(in_features)
        self.drop = torch.nn.Dropout(drop)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        trunc_normal_(self.linear.weight, std=std)
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(self.drop(self.batch_norm(x)))


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attention_ratio=[2],
                 mlp_ratio=[2],
                 down_ops=[],
                 distillation=True,
                 drop_path=0):
        super().__init__()
        resolution = img_size // patch_size
        self.stages = []

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = Backbone(in_chans, embed_dim[0])

        down_ops.append([''])

        for i, (ed, kd, dpth, nh, ar, mr,
                do) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attention_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.stages.append(
                    Residual(
                        Attention(
                            dim=ed,
                            key_dim=kd,
                            num_attention_heads=nh,
                            attention_ratio=ar,
                            resolution=resolution,
                        ),
                        drop_path))

                if mr > 0:
                    h = int(ed * mr)
                    self.stages.append(Residual(MLP(input_dim=ed, hidden_dim=h), drop_path))

            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.stages.append(
                    AttentionDownsample(input_dim=embed_dim[i],
                                        output_dim=embed_dim[i + 1],
                                        key_dim=do[1],
                                        num_attention_heads=do[2],
                                        attention_ratio=do[3],
                                        stride=do[5],
                                        resolution_in=resolution,
                                        resolution_out=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.stages.append(Residual(MLP(input_dim=embed_dim[i + 1], hidden_dim=h), drop_path))

        self.stages = torch.nn.Sequential(*self.stages)

        # Classifier head
        self.head = NormLinear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        if self.distillation:
            self.head_dist = NormLinear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.stages(x)
        x = x.mean(1)

        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)

        return x


def model_factory(embed_dim, key_dim, depth, num_heads, drop_path, weights, num_classes, distillation, pretrained):
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[key_dim] * 3,
        depth=depth,
        attention_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', key_dim, embed_dim[0] // key_dim, 4, 2, 2],
            ['Subsample', key_dim, embed_dim[1] // key_dim, 4, 2, 2],
        ],
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation)

    if pretrained:
        checkpoint_dict = torch.hub.load_state_dict_from_url(weights, map_location='cpu')
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        checkpoint_keys = list(checkpoint_dict.keys())
        for i, _ in enumerate(model_keys):
            if not (model_keys[i].startswith('head.linear') or model_keys[i].startswith('head_dist.linear')):
                model_dict[model_keys[i]] = checkpoint_dict[checkpoint_keys[i]]
        model.load_state_dict(model_dict)

    return model


def LeViT_128S(num_classes=1000, distillation=False, pretrained=False):
    return model_factory(**specification['LeViT_128S'],
                         num_classes=num_classes,
                         distillation=distillation,
                         pretrained=pretrained)


def LeViT_256(num_classes=1000, distillation=False, pretrained=False):
    return model_factory(**specification['LeViT_256'],
                         num_classes=num_classes,
                         distillation=distillation,
                         pretrained=pretrained)


def LeViT_384(num_classes=1000, distillation=False, pretrained=False):
    return model_factory(**specification['LeViT_384'],
                         num_classes=num_classes,
                         distillation=distillation,
                         pretrained=pretrained)


def lr_schedule_warmup(step, train_steps_epoch, init_lr):
    warmup_steps = train_steps_epoch * 3
    if step < warmup_steps:
        lr = init_lr / warmup_steps * step
    else:
        lr = init_lr
    return lr


def get_estimator(batch_size=64,
                  epochs=40,
                  data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  log_steps=100):

    train_data, eval_data = cifair10.load_data(data_dir)

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Resize(image_in="x", image_out="x", height=224, width=224),
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x"),
            Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.05)
        ])

    model = fe.build(model_fn=lambda: LeViT_384(num_classes=10, pretrained=True), optimizer_fn="adam")

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce", mode="train")
    ])

    init_lr = 1e-2 / 64 * batch_size

    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_schedule_warmup(
                step, train_steps_epoch=np.ceil(len(train_data) / batch_size), init_lr=init_lr)),
        4:
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(
                epoch, cycle_length=epochs - 3, init_lr=init_lr, min_lr=init_lr / 100, start=4))
    }

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=model_dir, metric="accuracy", save_best_mode="max"),
        EpochScheduler(lr_schedule)
    ]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch,
                             log_steps=log_steps)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()