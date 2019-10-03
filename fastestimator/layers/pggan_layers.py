import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Add

def nf(stage): 
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

class EqualizedLRDense(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[int(input_shape[-1]), self.units],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
            trainable=True
        )
        fan_in = np.prod(input_shape[-1])
        self.wscale = tf.constant(np.float32(np.sqrt(2) / np.sqrt(fan_in)))

    def call(self, input):
        return tf.matmul(input, self.w) * self.wscale


class EqualizedLRConv2D(Conv2D):
    def __init__(self, filters, gain=np.sqrt(2), kernel_size=3, strides=(1, 1), padding="same"):
        super().__init__(filters=filters, 
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        self.gain = gain

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            if input_shape.dims[channel_axis].value is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
        super().build(input_shape)
        print(input_shape)
        input_dim = int(input_shape[channel_axis])
        fan_in = np.prod(input_shape[1:])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))


    def call(self, input):
        return super().call(input) * self.wscale

class ApplyBias(Layer):
    def build(self, input_shape):
        self.b = self.add_weight(
            shape=input_shape[1],
            initializer='zeros',
            trainable=True
        )
        def call(self, input):
            # \NOTE(JP): The original code uses "tied" bias.
            if len(input.shape) == 2:
                return input + self.b
            else:
                return input + tf.reshape(self.b, [1, -1, 1, 1])

class FadeIn(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(initial_value=0.0, dtype='float32', trainable=False)

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

class PixelNormalization(Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        def call(self, input):
            return input * tf.math.rsqrt(tf.reduce_mean(tf.square(input), axis=1, keepdims=True) + self.eps)

class MiniBatchStd(Layer):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def call(self, input):
        group_size = tf.minimum(self.group_size, tf.shape(input)[0])
        s = input.shape
        y = tf.reshape(input, [group_size, -1, s[1], s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.redcue_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
        y = tf.cast(y, input.dtype)
        y = tf.tile(y, [self.group_size, 1, s[2], s[3]])
        return tf.concat([input, y], axis=1)
