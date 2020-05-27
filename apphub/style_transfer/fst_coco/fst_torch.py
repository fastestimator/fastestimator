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
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from torchvision import models

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data import mscoco
from fastestimator.layers.pytorch import Cropping2D
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver
from fastestimator.util import traceable


@traceable()
class ExtractVGGFeatures(TensorOp):
    def __init__(self, inputs, outputs, device, mode=None):
        super().__init__(inputs, outputs, mode)
        self.vgg = LossNet()
        self.vgg.to(device)

    def forward(self, data, state):
        return self.vgg(data)


@traceable()
class StyleContentLoss(TensorOp):
    def __init__(self, style_weight, content_weight, tv_weight, inputs, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.average_loss = average_loss

    def calculate_style_recon_loss(self, y_true, y_pred):
        y_true_gram = self.calculate_gram_matrix(y_true)
        y_pred_gram = self.calculate_gram_matrix(y_pred)
        y_diff_gram = y_pred_gram - y_true_gram
        y_norm = torch.sqrt(torch.sum(y_diff_gram * y_diff_gram, dim=(1, 2)))
        return y_norm

    def calculate_feature_recon_loss(self, y_true, y_pred):
        y_diff = y_pred - y_true
        num_elts = torch.tensor(y_diff.shape[1] * y_diff.shape[2] * y_diff.shape[3], dtype=torch.float32)
        y_diff_norm = torch.sum(y_diff * y_diff, dim=(1, 2, 3)) / num_elts
        return y_diff_norm

    def calculate_gram_matrix(self, x):
        num_elts = torch.tensor(x.shape[1] * x.shape[2] * x.shape[3], dtype=torch.float32)
        gram_matrix = torch.einsum('bcij,bdij->bcd', x, x)
        gram_matrix /= num_elts
        return gram_matrix

    def calculate_total_variation(self, y_pred):
        return (torch.sum(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]), dim=[1, 2, 3]) +
                torch.sum(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]), dim=[1, 2, 3]))

    def forward(self, data, state):
        y_pred, y_style, y_content, image_out = data

        style_loss = [self.calculate_style_recon_loss(a, b) for a, b in zip(y_style['style'], y_pred['style'])]
        style_loss = torch.stack(style_loss, dim=0).sum(dim=0)
        style_loss *= self.style_weight

        content_loss = [
            self.calculate_feature_recon_loss(a, b) for a, b in zip(y_content['content'], y_pred['content'])
        ]
        content_loss = torch.stack(content_loss, dim=0).sum(dim=0)
        content_loss *= self.content_weight

        total_variation_reg = self.calculate_total_variation(image_out)
        total_variation_reg *= self.tv_weight
        loss = style_loss + content_loss + total_variation_reg

        if self.average_loss:
            loss = reduce_mean(loss)

        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.cropping2d = Cropping2D(cropping=2)
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x0 = self.cropping2d(x)
        x = self.layers(x)
        x = x + x0
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4, apply_relu=True):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.apply_relu = apply_relu

        normal_(self.conv_layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.instance_norm(x)

        if self.apply_relu:
            x = fn.relu(x)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, output_size=(128, 128), padding=1):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.output_size = output_size

        normal_(self.conv_layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.conv_layer(x, output_size=self.output_size)
        x = self.instance_norm(x)
        x = fn.relu(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels)

        normal_(self.conv_layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.instance_norm(x)
        x = fn.relu(x)
        return x


class StyleTransferNet(nn.Module):
    """Style Transfer Network Architecture
    """
    def __init__(self) -> None:
        super().__init__()
        self.reflection_padding = nn.ReflectionPad2d(40)
        self.network_block = nn.Sequential(ConvBlock(3, 32),
                                           Downsample(32, 64),
                                           Downsample(64, 128),
                                           ResidualBlock(128, 128),
                                           ResidualBlock(128, 128),
                                           ResidualBlock(128, 128),
                                           ResidualBlock(128, 128),
                                           ResidualBlock(128, 128),
                                           Upsample(128, 64, output_size=(128, 128)),
                                           Upsample(64, 32, output_size=(256, 256)),
                                           ConvBlock(32, 3, apply_relu=False))

    def forward(self, x):
        x = self.reflection_padding(x)
        x = self.network_block(x)
        x = torch.tanh(x)
        return x


class LossNet(nn.Module):
    """Creates the network to compute the style loss.
    This network outputs a dictionary with outputs values for style and content.
    """
    def __init__(self) -> None:
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.layer1 = nn.Sequential(*list(vgg16.features.children())[:4])
        self.layer2 = nn.Sequential(*list(vgg16.features.children())[4:9])
        self.layer3 = nn.Sequential(*list(vgg16.features.children())[9:16])
        self.layer4 = nn.Sequential(*list(vgg16.features.children())[16:23])

    def forward(self, x):
        x_relu1 = self.layer1(x)
        x_relu2 = self.layer2(x_relu1)
        x_relu3 = self.layer3(x_relu2)
        x_relu4 = self.layer4(x_relu3)

        return {"style": [x_relu1, x_relu2, x_relu3, x_relu4], "content": [x_relu3]}


def get_estimator(batch_size=4,
                  epochs=2,
                  max_train_steps_per_epoch=None,
                  log_steps=100,
                  style_weight=5.0,
                  content_weight=1.0,
                  tv_weight=1e-4,
                  save_dir=tempfile.mkdtemp(),
                  style_img_path='Vassily_Kandinsky,_1913_-_Composition_7.jpg',
                  data_dir=None):
    train_data, _ = mscoco.load_data(root_dir=data_dir, load_bboxes=False, load_masks=False, load_captions=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    style_img = cv2.imread(style_img_path)
    assert style_img is not None, "cannot load the style image, please go to the folder with style image"
    style_img = cv2.resize(style_img, (256, 256))
    style_img = (style_img.astype(np.float32) - 127.5) / 127.5
    style_img_t = np.expand_dims(style_img, axis=0)
    style_img_t = np.transpose(style_img_t, (0, 3, 1, 2))
    style_img_t = torch.from_numpy(style_img_t).to(device)

    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
            Resize(height=256, width=256, image_in="image", image_out="image"),
            ChannelTranspose(inputs="image", outputs="image")
        ])

    model = fe.build(model_fn=StyleTransferNet,
                     model_name="style_transfer_net",
                     optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-3))

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="image_out"),
        ExtractVGGFeatures(inputs=lambda: style_img_t, outputs="y_style", device=device),
        ExtractVGGFeatures(inputs="image", outputs="y_content", device=device),
        ExtractVGGFeatures(inputs="image_out", outputs="y_pred", device=device),
        StyleContentLoss(style_weight=style_weight,
                         content_weight=content_weight,
                         tv_weight=tv_weight,
                         inputs=('y_pred', 'y_style', 'y_content', 'image_out'),
                         outputs='loss'),
        UpdateOp(model=model, loss_name="loss")
    ])

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=ModelSaver(model=model, save_dir=save_dir, frequency=1),
                             epochs=epochs,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             log_steps=log_steps)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
