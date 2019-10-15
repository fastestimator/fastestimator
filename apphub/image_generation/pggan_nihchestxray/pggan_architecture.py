import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, layers

fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
fmap_max = 512  # Maximum number of feature maps in any layer.

tf.random.set_seed(1000)


def nf(stage):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class FadeIn(layers.Add):
    def __init__(self, fade_in_alpha, **kwargs):
        super().__init__(**kwargs)
        self.fade_in_alpha = fade_in_alpha

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = ((1.0 - self.fade_in_alpha) * inputs[0]) + (self.fade_in_alpha * inputs[1])
        return output


class PixelNormalization(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


class MiniBatchStd(layers.Layer):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def call(self, x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = x.shape  # [NHWC]
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC]
        y = tf.reduce_mean(tf.square(y), axis=0)  #[MHWC]
        y = tf.sqrt(y + 1e-8)  # [MHWC]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.tile(y, [self.group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([x, y], axis=-1)


class EqualizedLRDense(layers.Layer):
    def __init__(self, units, gain=np.sqrt(2)):
        super().__init__()
        self.units = units
        self.gain = gain

    def build(self, input_shape):
        self.w = self.add_weight(shape=[int(input_shape[-1]), self.units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
                                 trainable=True)
        fan_in = np.prod(input_shape[-1])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, x):
        return tf.matmul(x, self.w) * self.wscale


class EqualizedLRConv2D(layers.Conv2D):
    def __init__(self, filters, gain=np.sqrt(2), kernel_size=3, strides=(1, 1), padding="same"):
        super().__init__(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.float32(np.prod(self.kernel.shape[:-1]))
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, x):
        return super().call(x) * self.wscale


class ApplyBias(layers.Layer):
    def build(self, input_shape):
        self.b = self.add_weight(shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, x):
        return x + self.b


def block_G(res, latent_dim=512, initial_resolution=2):
    if res == initial_resolution:
        x0 = layers.Input(shape=latent_dim)
        x = PixelNormalization()(x0)

        x = EqualizedLRDense(units=nf(res - 1) * 16, gain=np.sqrt(2) / 4)(x)
        x = tf.reshape(x, [-1, 4, 4, nf(res - 1)])
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)

        x = EqualizedLRConv2D(filters=nf(res - 1))(x)
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)
    else:
        x0 = layers.Input(shape=(2**(res - 1), 2**(res - 1), nf(res - 2)))
        x = layers.UpSampling2D()(x0)
        for _ in range(2):
            x = EqualizedLRConv2D(filters=nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = PixelNormalization()(x)
    return Model(inputs=x0, outputs=x, name="g_block_%dx%d" % (2**res, 2**res))


def torgb(res, num_channels=3):  # res = 2..resolution_log2
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    x = EqualizedLRConv2D(filters=num_channels, kernel_size=1, gain=1.0)(x0)
    x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="to_rgb_%dx%d" % (2**res, 2**res))


def build_G(fade_in_alpha, latent_dim=512, initial_resolution=2, target_resolution=10, num_channels=3):
    x0 = layers.Input(shape=latent_dim)
    curr_g_block = block_G(initial_resolution, latent_dim, initial_resolution)
    curr_to_rgb_block = torgb(initial_resolution, num_channels)
    images_out = curr_g_block(x0)
    images_out = curr_to_rgb_block(images_out)
    model_list = list()
    gen_block_list = list()

    mdl = Model(inputs=x0, outputs=images_out)
    model_list.append(mdl)
    gen_block_list.append(curr_g_block)
    prev_to_rgb_block = curr_to_rgb_block

    for res in range(initial_resolution + 1, target_resolution + 1):
        curr_g_block = block_G(res, latent_dim, initial_resolution)
        curr_to_rgb_block = torgb(res, num_channels)

        prev_images = x0
        for g in gen_block_list:
            prev_images = g(prev_images)

        curr_images = curr_g_block(prev_images)
        curr_images = curr_to_rgb_block(curr_images)

        prev_images = prev_to_rgb_block(prev_images)
        prev_images = layers.UpSampling2D(name="upsample_%dx%d" % (2**res, 2**res))(prev_images)

        images_out = FadeIn(fade_in_alpha=fade_in_alpha,
                            name="fade_in_%dx%d" % (2**res, 2**res))([prev_images, curr_images])
        mdl = Model(inputs=x0, outputs=images_out)
        model_list.append(mdl)
        gen_block_list.append(curr_g_block)
        prev_to_rgb_block = curr_to_rgb_block

    # build final model
    x = x0
    for g in gen_block_list:
        x = g(x)
    x = curr_to_rgb_block(x)
    final_mdl = Model(inputs=x0, outputs=x)
    model_list.append(final_mdl)
    return model_list


def fromrgb(res, num_channels=3):
    x0 = layers.Input(shape=(2**res, 2**res, num_channels))
    x = EqualizedLRConv2D(filters=nf(res - 1), kernel_size=1)(x0)
    x = ApplyBias()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return Model(inputs=x0, outputs=x, name="from_rgb_%dx%d" % (2**res, 2**res))


def block_D(res, initial_resolution, mbstd_group_size=4):
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    if res > initial_resolution:
        x = x0
        for i in range(2):
            x = EqualizedLRConv2D(filters=nf(res - (i + 1)))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.AveragePooling2D()(x)
    else:
        if mbstd_group_size > 1:
            x = MiniBatchStd(mbstd_group_size)(x0)
            x = EqualizedLRConv2D(filters=nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            x = layers.Flatten()(x)
            x = EqualizedLRDense(units=nf(res - 2))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = EqualizedLRDense(units=1, gain=1.0)(x)
            x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="d_block_%dx%d" % (2**res, 2**res))


def build_D(fade_in_alpha, mbstd_group_size=4, initial_resolution=2, target_resolution=10, num_channels=3):
    model_list = list()
    disc_block_list = list()
    for res in range(initial_resolution, target_resolution + 1):
        x0 = layers.Input(shape=(2**res, 2**res, num_channels))
        curr_from_rgb = fromrgb(res, num_channels)
        curr_D_block = block_D(res, initial_resolution, mbstd_group_size)
        x = curr_from_rgb(x0)
        x = curr_D_block(x)
        if res > 2:
            x_ds = layers.AveragePooling2D(name="downsample_%dx%d" % (2**res, 2**res))(x0)
            x_ds = prev_from_rgb(x_ds)
            x = FadeIn(fade_in_alpha=fade_in_alpha, name="fade_in_%dx%d" % (2**res, 2**res))([x_ds, x])
            for prev_d in disc_block_list[::-1]:
                x = prev_d(x)
        disc_block_list.append(curr_D_block)
        prev_from_rgb = curr_from_rgb
        mdl = Model(inputs=x0, outputs=x)
        model_list.append(mdl)
    return model_list
