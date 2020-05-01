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
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import kaiming_normal_ as he_normal
from torchvision import models

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data import cub200
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, ReadMat, ShiftScaleRotate
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage, Reshape
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy, Dice


class ReduceLoss(TensorOp):
    """TensorOp to average loss for a batch"""
    def forward(self, data, state):
        return reduce_mean(data)


class UncertaintyLossNet(nn.Module):
    """Creates Uncertainty weighted loss model https://arxiv.org/abs/1705.07115
    """
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        loss = torch.exp(-self.w1) * x[0] + self.w1 + torch.exp(-self.w2) * x[1] + self.w2
        return loss


class Upsample2D(nn.Module):
    """Upsampling Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
                                      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        for l in self.upsample:
            if isinstance(l, nn.Conv2d):
                he_normal(l.weight.data)

    def forward(self, x):
        return self.upsample(x)


class DecBlock(nn.Module):
    """Decoder Block"""
    def __init__(self, upsample_in_ch, conv_in_ch, out_ch):
        super().__init__()
        self.upsample = Upsample2D(upsample_in_ch, out_ch)
        self.conv_layers = nn.Sequential(nn.Conv2d(conv_in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True))

        for l in self.conv_layers:
            if isinstance(l, nn.Conv2d):
                he_normal(l.weight.data)

    def forward(self, x_up, x_down):
        x = self.upsample(x_up)
        x = torch.cat([x, x_down], 1)
        x = self.conv_layers(x)
        return x


class ResUnet50(nn.Module):
    """Network Architecture"""
    def __init__(self, num_classes=200):
        super().__init__()
        base_model = models.resnet50(pretrained=True)

        self.enc1 = nn.Sequential(*list(base_model.children())[:3])
        self.input_pool = list(base_model.children())[3]
        self.enc2 = nn.Sequential(*list(base_model.children())[4])
        self.enc3 = nn.Sequential(*list(base_model.children())[5])
        self.enc4 = nn.Sequential(*list(base_model.children())[6])
        self.enc5 = nn.Sequential(*list(base_model.children())[7])
        self.fc = nn.Linear(2048, num_classes)

        self.dec6 = DecBlock(2048, 1536, 512)
        self.dec7 = DecBlock(512, 768, 256)
        self.dec8 = DecBlock(256, 384, 128)
        self.dec9 = DecBlock(128, 128, 64)
        self.dec10 = Upsample2D(64, 2)
        self.mask = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        x_e1 = self.enc1(x)
        x_e1_1 = self.input_pool(x_e1)
        x_e2 = self.enc2(x_e1_1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)
        x_e5 = self.enc5(x_e4)

        x_label = fn.max_pool2d(x_e5, kernel_size=x_e5.size()[2:])
        x_label = x_label.view(x_label.shape[0], -1)
        x_label = self.fc(x_label)
        x_label = torch.softmax(x_label, dim=-1)

        x_d6 = self.dec6(x_e5, x_e4)
        x_d7 = self.dec7(x_d6, x_e3)
        x_d8 = self.dec8(x_d7, x_e2)
        x_d9 = self.dec9(x_d8, x_e1)
        x_d10 = self.dec10(x_d9)
        x_mask = self.mask(x_d10)
        x_mask = torch.sigmoid(x_mask)
        return x_label, x_mask


def get_estimator(batch_size=8,
                  epochs=25,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir=None):
    # load CUB200 dataset.
    train_data = cub200.load_data(root_dir=data_dir)
    eval_data = train_data.split(0.3)
    test_data = eval_data.split(0.5)

    #step 1, pipeline
    pipeline = fe.Pipeline(
        batch_size=batch_size,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        ops=[
            ReadImage(inputs="image", outputs="image", parent_path=train_data.parent_path),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
            ReadMat(file='annotation', keys="seg", parent_path=train_data.parent_path),
            Delete(keys="annotation"),
            LongestMaxSize(max_size=512, image_in="image", image_out="image", mask_in="seg", mask_out="seg"),
            PadIfNeeded(min_height=512,
                        min_width=512,
                        image_in="image",
                        image_out="image",
                        mask_in="seg",
                        mask_out="seg",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0),
            ShiftScaleRotate(image_in="image",
                             mask_in="seg",
                             image_out="image",
                             mask_out="seg",
                             mode="train",
                             shift_limit=0.2,
                             rotate_limit=15.0,
                             scale_limit=0.2,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0,
                             mask_value=0),
            Sometimes(HorizontalFlip(image_in="image", mask_in="seg", image_out="image", mask_out="seg", mode="train")),
            ChannelTranspose(inputs="image", outputs="image"),
            Reshape(shape=(1, 512, 512), inputs="seg", outputs="seg")
        ])

    #step 2, network
    resunet50 = fe.build(model_fn=ResUnet50,
                         model_name="resunet50",
                         optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-4))
    uncertainty = fe.build(model_fn=UncertaintyLossNet,
                           model_name="uncertainty",
                           optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-5))

    network = fe.Network(ops=[
        ModelOp(inputs='image', model=resunet50, outputs=["label_pred", "mask_pred"]),
        CrossEntropy(inputs=["label_pred", "label"], outputs="cls_loss", form="sparse", average_loss=False),
        CrossEntropy(inputs=["mask_pred", "seg"], outputs="seg_loss", form="binary", average_loss=False),
        ModelOp(inputs=["cls_loss", "seg_loss"], model=uncertainty, outputs="total_loss"),
        ReduceLoss(inputs="total_loss", outputs="total_loss"),
        UpdateOp(model=resunet50, loss_name="total_loss"),
        UpdateOp(model=uncertainty, loss_name="total_loss")
    ])

    #step 3, estimator
    traces = [
        Accuracy(true_key="label", pred_key="label_pred"),
        Dice(true_key="seg", pred_key='mask_pred'),
        BestModelSaver(model=resunet50, save_dir=save_dir, metric="total_loss", save_best_mode="min"),
        LRScheduler(model=resunet50, lr_fn=lambda step: cosine_decay(step, cycle_length=13200, init_lr=1e-4))
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=traces,
                             epochs=epochs,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             log_steps=500)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
