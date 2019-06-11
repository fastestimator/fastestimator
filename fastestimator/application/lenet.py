"""LeNet tf.keras Definition."""

import tensorflow as tf
from tensorflow.keras.layers import ELU, Conv2D, Dense, Flatten, Input, MaxPool2D


# pylint: disable=invalid-name
def LeNet(input_name, output_name, input_shape=(28, 28, 1), output_class=10, random_seed=None):
    """Produces LeNet architecture.

    Args:
        input_name : Name of input feature.
        output_name : Name of output feature.
        input_shape: Shape of input feature.
        output_class: Number of output classes.

    Returns:
        tf.keras.models.Model object.

    """

    kernel_seed = random_seed
    conv_args = {'kernel_size': (5, 5),
                 'padding': 'same',
                 'kernel_initializer': tf.keras.initializers.glorot_normal(seed=kernel_seed)}
    dense_args = {'kernel_initializer': tf.keras.initializers.glorot_normal(seed=kernel_seed)}

    img_input = Input(input_shape, name=input_name)
    x = Conv2D(filters=32, **conv_args)(img_input)
    x = ELU()(x)
    x = MaxPool2D()(x)
    x = Conv2D(filters=64, **conv_args)(x)
    x = ELU()(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(256, **dense_args)(x)
    x = ELU()(x)
    x = Dense(128, **dense_args)(x)
    x = ELU()(x)
    outputs = Dense(output_class, activation='softmax', name=output_name, **dense_args)(x)
    model = tf.keras.Model(inputs=img_input, outputs=outputs)

    return model
