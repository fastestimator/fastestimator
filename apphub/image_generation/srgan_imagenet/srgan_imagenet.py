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
"""SRGAN training using ImageNet data set."""
import os
import pdb
import tempfile

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU
from tensorflow.keras.models import Model

import fastestimator as fe
from fastestimator.dataset import srgan
from fastestimator.layers.sub_pixel_conv_2d import SubPixelConv2D
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.tensorop import Loss, ModelOp, Rescale
from fastestimator.schedule.lr_scheduler import LRSchedule
from fastestimator.trace import LRController, ModelSaver


class DLoss(Loss):
    """Compute discrimator loss."""
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, data, state):
        true, fake = data
        real_loss = self.cross_entropy(tf.ones_like(true), true)
        fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss


class GLoss(Loss):
    """Compute generator loss."""
    def __init__(self, inputs, outputs=None, mode=None, vgg_content=False, input_shape=(96, 96, 3)):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.vgg_content = vgg_content
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss = tf.losses.MeanSquaredError(reduction='none')
        # to calculate vgg loss
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        self.vgg_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

    def vgg_preprocess(self, x):
        x = 255.0 * (0.5 * (x + 1.0))
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])
        x = x - mean
        x = x[:, :, :, ::-1]
        return x

    def vgg_forward(self, superres, highres):
        superres = self.vgg_preprocess(superres)
        highres = self.vgg_preprocess(highres)
        superres = self.vgg_model(superres)
        highres = self.vgg_model(highres)
        return superres, highres

    def forward(self, data, state):
        superres, highres, fake = data
        batch_size, _, _, _ = superres.shape
        if self.vgg_content:
            superres, highres = self.vgg_forward(superres, highres)
            superres = tf.reshape(superres, (batch_size, -1))
            highres = tf.reshape(highres, (batch_size, -1))
            mse_loss = 0.006 * self.mse_loss(highres, superres)

        else:
            superres = tf.reshape(superres, (batch_size, -1))
            highres = tf.reshape(highres, (batch_size, -1))
            mse_loss = self.mse_loss(highres, superres)
        fake_loss = self.cross_entropy(tf.ones_like(fake), fake)
        total_loss = mse_loss + 0.001 * fake_loss
        return total_loss, mse_loss, 0.001 * fake_loss


class MyLRSchedule(LRSchedule):
    """ lrschedule to modify lr for srgan.  """
    def schedule_fn(self, current_step_or_epoch, lr):
        if current_step_or_epoch <= 100000:
            lr = 0.0001
        elif current_step_or_epoch > 100000:
            lr = 0.00001
        return lr


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


def get_generator(input_shape):
    """  generator model.    """
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


def get_discriminator(input_shape):
    """  discriminator model .   """
    def d_block(layer_input, filters, strides=1, bn=True):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal')(layer_input)
        if bn:
            d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    # Input img
    d0 = Input(shape=input_shape)
    d1 = d_block(d0, 64, strides=1, bn=False)
    d2 = d_block(d1, 64, strides=2)
    d3 = d_block(d2, 128, strides=1)
    d4 = d_block(d3, 128, strides=2)
    d5 = d_block(d4, 256, strides=1)
    d6 = d_block(d5, 256, strides=2)
    d7 = d_block(d6, 512, strides=1)
    d8 = d_block(d7, 512, strides=2)

    d8 = Flatten()(d8)
    d9 = Dense(1024)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1)(d10)

    return Model(d0, validity)


#srgan/srresnet stipulate batch_size to be 16. The scripts has been executed on p2.8xlarge ec2 instance
# having 4 GPGPU. hence batch_size of 4 is passed
def get_estimator(batch_size=4,
                  epochs=200,
                  steps_per_epoch=1000,
                  validation_steps=None,
                  model_dir=tempfile.mkdtemp(),
                  imagenet_path=None,
                  srresnet_model_path=None):
    """Args:
        imagenet_path: folder path of ImageNet dataset, containing train and val subdirs .
        srresnet_model_path: srresnet model weights, srgan generator gets initialized with the weights.
    """

    assert imagenet_path is not None, 'Pass valid folder path of Imagenet dataset'
    assert srresnet_model_path is not None, 'srresnet model is needed to initialize srgan generator model'
    # Ensure ImageNet dataset is downloaded. Pass the folder contianing train and val subdirectories.
    # currently the script doesn't download the ImageNet data.
    train_csv, val_csv, path = srgan.load_data(path_imgnet=imagenet_path)

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
    model_gen = fe.build(model_def=srresnet_model_path,
                         model_name="srgan_gen",
                         optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                         loss_name="mse_adv_loss",
                         custom_objects={'SubPixelConv2D': SubPixelConv2D})
    model_desc = fe.build(model_def=lambda: get_discriminator(input_shape=(96, 96, 3)),
                          model_name="srgan_desc",
                          optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                          loss_name="desc_loss")

    network = fe.Network(ops=[
        ModelOp(inputs='lowres', model=model_gen, outputs='superres'),
        ModelOp(inputs='superres', model=model_desc, outputs='pred_fake'),
        ModelOp(inputs='highres', model=model_desc, outputs='pred_true'),
        DLoss(inputs=("pred_true", "pred_fake"), outputs=("desc_loss", "real_loss", "fake_loss")),
        GLoss(inputs=('superres', 'highres', 'pred_fake'),
              outputs=("mse_adv_loss", "mse_loss", "adv_loss"),
              vgg_content=True)
    ])

    model_dir = os.path.join(path)
    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        traces=[
            ModelSaver(model_name="srgan_gen", save_dir=model_dir, save_best=True),
            ModelSaver(model_name="srgan_desc", save_dir=model_dir, save_best=True),
            LRController(model_name="srgan_gen", lr_schedule=MyLRSchedule(schedule_mode='step')),
            LRController(model_name="srgan_desc", lr_schedule=MyLRSchedule(schedule_mode='step'))
        ])

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
