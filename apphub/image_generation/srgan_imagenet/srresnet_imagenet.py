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
"""SRRESNET training using ImageNet data set."""

import os
import pdb
import tempfile

import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Input, PReLU
from tensorflow.keras.models import Model

import fastestimator as fe
from fastestimator.dataset import srgan
from fastestimator.layers.sub_pixel_conv_2d import SubPixelConv2D
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.tensorop import Loss, ModelOp, Rescale
from fastestimator.trace import ModelSaver


class LowresRescale(TensorOp):
    """Rescaling data according to

    Args:
        inputs: Name of the key in the dataset that is to be filtered.
        outputs: Name of the key to be created/used in the dataset to store the results.
        mode: mode that the filter acts on.
    """
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data /= 255
        return data


class PixelMeanSquaredError(Loss):
    """Calculate mean squared error loss averaged by number of pixels,

    Args:
        y_true: ground truth label key
        y_pred: prediction label key
        inputs: A tuple or list like: [<y_true>, <y_pred>]
        outputs: Where to store the computed loss value (not required under normal use cases)
        mode: 'train', 'eval', 'test', or None
        kwargs: Arguments to be passed along to the tf.losses constructor
    """
    def __init__(self, y_true=None, y_pred=None, inputs=None, outputs=None, mode=None, **kwargs):

        if 'reduction' in kwargs:
            raise KeyError("parameter 'reduction' not allowed")
        inputs = self.validate_loss_inputs(inputs, y_true, y_pred)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_obj = tf.losses.MeanSquaredError(reduction='none', **kwargs)

    def forward(self, data, state):
        true, pred = data
        batch_size, _, _, _ = true.shape
        true = tf.reshape(true, (batch_size, -1))
        pred = tf.reshape(pred, (batch_size, -1))
        loss = self.loss_obj(true, pred)
        return loss


def get_generator(input_shape):
    """ generator model."""
    nin = Input(input_shape)
    n = Conv2D(64, 9, 1, padding='SAME', kernel_initializer='he_normal')(nin)
    n = PReLU(shared_axes=[1, 2])(n)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2D(64, 3, 1, padding='SAME', use_bias=False, kernel_initializer='he_normal')(n)

        nn = BatchNormalization()(nn)
        nn = PReLU(shared_axes=[1, 2])(nn)

        nn = Conv2D(64, 3, 1, padding='SAME', use_bias=False, kernel_initializer='he_normal')(nn)
        nn = BatchNormalization()(nn)
        nn = Add()([n, nn])
        n = nn

    n = Conv2D(64, 3, 1, padding='SAME', use_bias=False, kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = Add()([n, temp])
    # B residual blacks end

    n = Conv2D(256, 3, 1, padding='SAME', kernel_initializer='he_normal')(n)
    n = SubPixelConv2D(upsample_factor=2, nchannels=64)(n)
    n = PReLU(shared_axes=[1, 2])(n)

    n = Conv2D(256, 3, 1, padding='SAME', kernel_initializer='he_normal')(n)
    n = SubPixelConv2D(upsample_factor=2, nchannels=64)(n)
    n = PReLU(shared_axes=[1, 2])(n)

    nn = Conv2D(3, 9, 1, padding='SAME', kernel_initializer='he_normal')(n)
    return Model(inputs=nin, outputs=nn, name="generator")


# Note: srgan/srresnet stipulate batch_size to be 16. The scripts has been executed on p3.8xlarge
# ec2 instance having 4 GPGPU. hence batch_size of 4 is passed
def get_estimator(batch_size=4, epochs=1000, steps_per_epoch=1000, validation_steps=None, model_dir=tempfile.mkdtemp(), imagenet_path=None):
    """Args:
        imagenet_path: folder path of ImageNet dataset, containing train and val subdirs .
    """

    assert imagenet_path is not None, 'Pass valid folder path of Imagenet dataset'
    # Ensure ImageNet dataset is downloaded. Pass the folder contianing train and val subdirectories.
    # currently the script doesn't download the ImageNet data.
    train_csv, val_csv, path = srgan.load_data(path_imgnet= imagenet_path)

    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "sr_tfrecords"),
        train_data=train_csv,
        validation_data=val_csv,
        ops=[ImageReader(inputs="lowres", outputs="lowres"), ImageReader(inputs="highres", outputs="highres")],
        compression="GZIP",
        write_feature=['lowres', 'highres'])

    pipeline = fe.Pipeline(
        max_shuffle_buffer_mb=3000,
        batch_size=batch_size,
        data=writer,
        ops=[
            LowresRescale(inputs='lowres', outputs='lowres'),
            Rescale(inputs='highres', outputs='highres'), ])

    # prepare model
    model = fe.build(model_def=lambda: get_generator(input_shape=(24, 24, 3)),
                     model_name="srresnet_gen",
                     optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                     loss_name="mse_loss")
    network = fe.Network(ops=[
        ModelOp(inputs='lowres', model=model, outputs='superres'),
        PixelMeanSquaredError(inputs=('superres', 'highres'), outputs="mse_loss")
    ])

    model_dir = os.path.join(path)
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs,
                             traces=[
                                 ModelSaver(model_name="srresnet_gen", save_dir=model_dir, save_best=True), ])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
