import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
fmap_max = 512  # Maximum number of feature maps in any layer.


def nf(stage):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class FadeIn(layers.Add):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(initial_value=0.0, dtype='float32', trainable=False)

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class PixelNormalization(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def call(self, input):
        return input * tf.math.rsqrt(tf.reduce_mean(tf.square(input), axis=1, keepdims=True) + self.eps)


class MiniBatchStd(layers.Layer):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def call(self, input):
        group_size = tf.minimum(self.group_size, tf.shape(input)[0])
        s = input.shape  # [NHWC]
        y = tf.reshape(input, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC]
        y = tf.cast(y, tf.float32)  # [GMHWC]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC]
        y = tf.reduce_mean(tf.square(y), axis=0)  #[MHWC]
        y = tf.sqrt(y + 1e-8)  # [MCHW]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.cast(y, input.dtype)  # [M111]
        y = tf.tile(y, [self.group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([input, y], axis=-1)


def block_G(res, latent_dim=512, num_channels=3, target_res=10):
    if res == 2:
        x0 = layers.Input(shape=(latent_dim, ))
        x = PixelNormalization()(x0)

        x = layers.Dense(units=nf(res - 1) * 16)(x)
        x = tf.reshape(x, [-1, 4, 4, nf(res - 1)])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)

        x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)
    else:
        x0 = layers.Input(shape=(2**(res - 1), 2**(res - 1), nf(res - 2)))
        x = layers.UpSampling2D()(x0)
        for _ in range(2):
            x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = PixelNormalization()(x)
    return Model(inputs=x0, outputs=x, name="g_block_%dx%d" % (2**res, 2**res))


def torgb(res, num_channels=3):  # res = 2..resolution_log2
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    x = layers.Conv2D(filters=num_channels, kernel_size=1, padding="same")(x0)
    return Model(inputs=x0, outputs=x, name="to_rgb_%dx%d" % (2**res, 2**res))


def build_G(latent_dim=512, initial_resolution=2, target_resolution=10):
    x0 = layers.Input(shape=(latent_dim, ))
    curr_g_block = block_G(initial_resolution)
    curr_to_rgb_block = torgb(initial_resolution)
    images_out = curr_g_block(x0)
    images_out = curr_to_rgb_block(images_out)

    model_list = list()
    gen_block_list = list()

    mdl = Model(inputs=x0, outputs=images_out)
    mdl.alpha = tf.convert_to_tensor(1.0)
    model_list.append(mdl)

    gen_block_list.append(curr_g_block)

    prev_g_block = curr_g_block
    prev_to_rgb_block = curr_to_rgb_block

    for res in range(3, target_resolution + 1):
        curr_g_block = block_G(res)
        curr_to_rgb_block = torgb(res)

        prev_images = x0
        for g in gen_block_list:
            prev_images = g(prev_images)

        curr_images = curr_g_block(prev_images)
        curr_images = curr_to_rgb_block(curr_images)

        prev_images = prev_to_rgb_block(prev_images)
        prev_images = layers.UpSampling2D(name="upsample_%dx%d" % (2**res, 2**res))(prev_images)

        fade_in = FadeIn(alpha=0.0, name="fade_in_%dx%d" % (2**res, 2**res))

        images_out = fade_in([curr_images, prev_images])
        mdl = Model(inputs=x0, outputs=images_out)
        mdl.alpha = fade_in.alpha
        model_list.append(mdl)
        gen_block_list.append(curr_g_block)

        prev_g_block = curr_g_block
        prev_to_rgb_block = curr_to_rgb_block

    # build final model
    final_output = x0
    x = x0
    for g in gen_block_list:
        x = g(x)
    x = curr_to_rgb_block(x)
    final_mdl = Model(inputs=x0, outputs=x)
    model_list.append(final_mdl)
    return model_list

G = build_G()
for i, g in enumerate(G):
    print("%dx%d" % (2**(i + 2), 2**(i + 2)))
    g.summary(100)

def fromrgb(res, num_channels=3):
    x0 = layers.Input(shape=(2**res, 2**res, num_channels))
    x = layers.Conv2D(filters=nf(res - 1), kernel_size=1, padding="same")(x0)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return Model(inputs=x0, outputs=x, name="from_rgb_%dx%d" % (2**res, 2**res))


def block_D(res, mbstd_group_size=4):
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    if res >= 3:
        x = x0
        for i in range(2):
            x = layers.Conv2D(filters=nf(res - (i + 1)), kernel_size=3, padding="same")(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.AveragePooling2D()(x)
    else:
        if mbstd_group_size > 1:
            x = MiniBatchStd(mbstd_group_size)(x0)
            x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            x = layers.Flatten()(x)
            x = layers.Dense(units=nf(res - 2))(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            x = layers.Dense(units=1)(x)
    return Model(inputs=x0, outputs=x, name="d_block_%dx%d" % (2**res, 2**res))


def build_D(target_resolution=10):
    model_list = list()
    disc_block_list = list()
    for res in range(2, target_resolution + 1):
        x0 = layers.Input(shape=(2**res, 2**res, 3))
        curr_from_rgb = fromrgb(res)
        curr_D_block = block_D(res)

        x = curr_from_rgb(x0)
        x = curr_D_block(x)

        if res > 2:
            x_ds = layers.AveragePooling2D(name="downsample_%dx%d" % (2**res, 2**res))(x0)
            x_ds = prev_from_rgb(x_ds)
            fade_in = FadeIn(alpha=0.0, name="fade_in_%dx%d" % (2**res, 2**res))
            x = fade_in([x_ds, x])
            for prev_d in disc_block_list[::-1]:
                x = prev_d(x)

            mdl = Model(inputs=x0, outputs=x)
            mdl.alpha = fade_in.alpha
            model_list.append(mdl)
        else:
            mdl = Model(inputs=x0, outputs=x)
            mdl.alpha = tf.convert_to_tensor(1.0)
            model_list.append(mdl)

        disc_block_list.append(curr_D_block)
        prev_from_rgb = curr_from_rgb
    return model_list


D = build_D()

for i, d in enumerate(D):
    print("%dx%d" % (2**(i + 2), 2**(i + 2)))
    d.summary(100)
