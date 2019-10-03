import tensorflow as tf
import numpy as np

from fastestimator.layers.pggan_layers import *


def block_G(x, res):
    if res == 2:
        x = PixelNormalization()(x)

        x = EqualizedLRDense(units=nf(res - 1) * 16)(x)
        x = tf.reshape(x, [-1, 4, 4, nf(res - 1)])
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)

        x = EqualizedLRConv2D(filters=nf(res - 1))(x)
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)
    else:
        x = layers.UpSampling2D()(x)
        for _ in range(2):
            x = EqualizedLRConv2D(filters=nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = PixelNormalization()(x)
    return x


def torgb(x, num_channels=3):  # res = 2..resolution_log2
    x = EqualizedLRConv2D(filters=num_channels, kernel_size=1, gain=1)(x)
    x = ApplyBias()(x)
    return x


def build_G(resolution=10):
    x0 = layers.Input(shape=(512, ))
    x = block_G(x0, 2)
    images_out = torgb(x)
    model_list = []
    mdl = tf.keras.Model(inputs=x0, outputs=images_out)
    mdl.alpha = tf.convert_to_tensor(1)
    model_list.append(mdl)
    for res in range(3, resolution + 1):
        x = block_G(x, res)
        img = torgb(x)
        images_out = layers.UpSampling2D()(images_out)
        fade_in_layer = FadeIn(alpha=0)
        images_out = fade_in_layer([images_out, img])
        mdl = tf.keras.Model(inputs=x0, outputs=images_out)
        mdl.alpha = fade_in_layer.alpha
        print(mdl.alpha)
        model_list.append(mdl)
    return model_list
