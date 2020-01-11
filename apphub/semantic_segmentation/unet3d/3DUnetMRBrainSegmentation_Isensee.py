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

import os
import tempfile

import nibabel as nib
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import UNet3D_Isensee
from fastestimator.dataset import brats
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ImageReader, NIBImageReader
from fastestimator.op.tensorop import Loss, ModelOp
from fastestimator.op.tensorop.augmentation import Augmentation3D
from fastestimator.op.tensorop.loss import WeightedDiceLoss
from fastestimator.trace import LRController, ModelSaver


class RandomImagePatches(TensorOp):
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 image_shape=(144, 144, 144),
                 patch_shape=(64, 64, 64),
                 patch_start_offset=16):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.nlabels = 3
        self.labels = [1, 2, 4]
        self.image_shape = tf.constant(image_shape, dtype=tf.dtypes.int32)
        self.patch_start_offset = tf.constant(patch_start_offset, dtype=tf.dtypes.int32)
        self.patch_shape = tf.constant(patch_shape, dtype=tf.dtypes.int32)
        self.modality_images, self.seg_mask = None, None

    def handle_padding_case(self, patch):
        pad_before = tf.where(patch < 0, tf.abs(patch), 0)
        pad_after = tf.where((patch + self.patch_shape) > self.image_shape,
                             tf.abs((patch + self.patch_shape) - self.image_shape),
                             0)
        padding = tf.stack([pad_before, pad_after], axis=1)
        no_padding = [[0, 0]]  # no padding in the batch dimension and the next dimention
        padding = tf.concat([no_padding, padding], axis=0)
        data_pad = tf.pad(self.modality_images, padding, mode='CONSTANT', constant_values=0)
        truth_pad = tf.pad(self.segmask_multi_class, padding, mode='CONSTANT', constant_values=0)
        patch = patch + pad_before
        data_patch = data_pad[...,
                              patch[0]:patch[0] + self.patch_shape[0],
                              patch[1]:patch[1] + self.patch_shape[1],
                              patch[2]:patch[2] + self.patch_shape[2]]
        truth_patch = truth_pad[...,
                                patch[0]:patch[0] + self.patch_shape[0],
                                patch[1]:patch[1] + self.patch_shape[1],
                                patch[2]:patch[2] + self.patch_shape[2]]
        return data_patch, truth_patch

    def handle_nopaddding_case(self, patch):
        data_pad = self.modality_images
        truth_pad = self.segmask_multi_class
        data_patch = data_pad[...,
                              patch[0]:patch[0] + self.patch_shape[0],
                              patch[1]:patch[1] + self.patch_shape[1],
                              patch[2]:patch[2] + self.patch_shape[2]]
        truth_patch = truth_pad[...,
                                patch[0]:patch[0] + self.patch_shape[0],
                                patch[1]:patch[1] + self.patch_shape[1],
                                patch[2]:patch[2] + self.patch_shape[2]]
        return data_patch, truth_patch

    def get_image_patch(self, patch):
        cond1 = tf.reduce_any(patch < 0)
        cond2 = tf.reduce_any(patch + self.patch_shape > self.image_shape)
        cond = tf.logical_or(cond1, cond2)
        data_patch, truth_patch = tf.cond(cond,lambda: self.handle_padding_case(patch),
                                          lambda:self.handle_nopaddding_case(patch))
        return data_patch, truth_patch

    def forward(self, data, state):
        self.modality_images, self.seg_mask = data
        cls_1 = tf.where(self.seg_mask[0] == self.labels[0], 1, 0)
        cls_2 = tf.where(self.seg_mask[0] == self.labels[1], 1, 0)
        cls_4 = tf.where(self.seg_mask[0] == self.labels[2], 1, 0)
        self.segmask_multi_class = tf.stack([cls_1, cls_2, cls_4])

        # We randomly select the start of the patches
        random_start_offset = tf.negative(
            tf.random.uniform(shape=(3, ), minval=0, maxval=self.patch_start_offset + 1, dtype=tf.dtypes.int32))

        start = random_start_offset
        stop = self.image_shape + random_start_offset
        step = self.patch_shape
        x,y,z = tf.meshgrid( tf.range(start[0],stop[0],step[0]),  tf.range(start[1],stop[1],step[1]),
                                        tf.range(start[2],stop[2],step[2]),indexing='ij')
        patches = tf.stack([x, y, z], axis=0)
        patches = tf.transpose(tf.reshape(patches, (3, -1)))

        data_stack,truth_stack = tf.map_fn(self.get_image_patch,patches,
                                           dtype=(tf.dtypes.float32, tf.dtypes.int32), back_prop=False)
        mask = tf.map_fn(lambda tr: tf.reduce_any(tf.not_equal(tr, 0)), truth_stack, dtype=tf.bool, back_prop=False)
        data_stack = tf.boolean_mask(data_stack, mask)
        truth_stack = tf.boolean_mask(truth_stack, mask)

        return data_stack, truth_stack


class SplitMaskLabelwise(TensorOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.nlabels = 3
        self.labels = [1, 2, 4]

    def forward(self, data, state):
        seg_mask = data
        cls_1 = tf.where(seg_mask[0] == self.labels[0], 1, 0)
        cls_2 = tf.where(seg_mask[0] == self.labels[1], 1, 0)
        cls_4 = tf.where(seg_mask[0] == self.labels[2], 1, 0)
        segmask_multi_label = tf.stack([cls_1, cls_2, cls_4])
        return segmask_multi_label


def get_estimator(batch_size=1,
                  epochs=500,
                  steps_per_epoch=128,
                  model_dir=tempfile.mkdtemp(),
                  path_brats=os.path.join(os.getenv('HOME'), 'fastestimator_data', 'BraTs')):
    """Args:
        path_brats: folder path of BraTs 2018 dataset, containing data subdir inturn having LGG and HGG data.
        Expected folder structure path_brats
        path_brats/
        |----------data/
                   |----LGG/
                   |----HGG/
    """
    assert path_brats is not None, 'Pass valid folder path of BraTs 2018 dataset'
    # Ensure Brats 2018 dataset is downloaded. Pass the folder contianing train and val subdirectories.
    # currently the script doesn't download the BraTs data.
    train_csv, val_csv, path_brats = brats.load_data(path_brats=path_brats,
                                               resized_img_shape=(128,128,128), bias_correction=False)
    writer = fe.RecordWriter(
        save_dir=os.path.join(path_brats, "tfrecords"),
        train_data=train_csv,
        validation_data=val_csv,
        ops=[
            NIBImageReader(inputs="mod_img", outputs="mod_img"), NIBImageReader(inputs="seg_mask", outputs="seg_mask")
        ],
        compression="GZIP",
        write_feature=['mod_img', 'seg_mask'])

    pipeline = fe.Pipeline(
        data=writer,
        batch_size=batch_size,
        ops=[
            SplitMaskLabelwise(inputs="seg_mask", outputs="seg_mask"),
            Augmentation3D(inputs=("mod_img", "seg_mask"), outputs=("mod_img", "seg_mask"), mode='train')
        ])

    model_unet3d_is = fe.build(model_def=lambda: UNet3D_Isensee(input_shape=(4, 64, 64, 64)),
                               model_name="brats_unet3d_is",
                               optimizer=tf.optimizers.Adam(learning_rate=5e-4),
                               loss_name="wdice_loss")

    network = fe.Network(ops=[
        ModelOp(inputs="mod_img", model=model_unet3d_is, outputs="pred_seg_mask"),
        WeightedDiceLoss(inputs=("seg_mask", "pred_seg_mask"), outputs=("wdice_loss"))
    ])
    model_dir = path_brats
    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        traces=[
            ModelSaver(model_name="brats_unet3d_is", save_dir=model_dir, save_best=True),
            LRController(model_name="brats_unet3d_is",
                         reduce_patience=10,
                         reduce_factor=0.5,
                         reduce_on_eval=True,
                         min_lr=1e-07)
        ],
        log_steps=steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
