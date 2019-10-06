import tensorflow as tf
from fastestimator.layers import InstanceNormalization, ReflectionPadding2D


def _residual_block(x0, num_filter, kernel_size=(3, 3), strides=(1, 1)):
    initializer = tf.random_normal_initializer(0., 0.02)
    x0_cropped = tf.keras.layers.Cropping2D(cropping=2)(x0)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x0)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.Add()([x, x0_cropped])
    return x


def _conv_block(x0, num_filter, kernel_size=(9, 9), strides=(1, 1), padding="same", apply_relu=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    if apply_relu:
        x = tf.keras.layers.ReLU()(x)
    return x


def _upsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters=num_filter,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _downsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def styleTransferNet(input_shape=(256, 256, 3), num_resblock=5):
    """Creates the style Transfer Network.

    Args:
        input_shape (tuple, optional): shape of input image. Defaults to (256, 256, 3).
        num_resblock (int, optional): number of resblocks for the network. Defaults to 5.

    Returns:
        'Model' object: style Transfer Network.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    x = ReflectionPadding2D(padding=(40, 40))(x0)
    x = _conv_block(x, num_filter=32)
    x = _downsample(x, num_filter=64)
    x = _downsample(x, num_filter=128)

    for _ in range(num_resblock):
        x = _residual_block(x, num_filter=128)

    x = _upsample(x, num_filter=64)
    x = _upsample(x, num_filter=32)
    x = _conv_block(x, num_filter=3, apply_relu=False)
    x = tf.keras.layers.Activation("tanh")(x)
    return tf.keras.Model(inputs=x0, outputs=x)


def lossNet(input_shape=(256, 256, 3),
            styleLayers=["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"],
            contentLayers=["block3_conv3"]):
    """Creates the network to compute the style loss.
    This network outputs a dictionary with outputs values for style and content, based on a list of layers from VGG16
    for each.

    Args:
        input_shape (tuple, optional): shape of input image. Defaults to (256, 256, 3).
        styleLayers (list, optional): list of style layers from VGG16. Defaults to ["block1_conv2", "block2_conv2",
        "block3_conv3", "block4_conv3"].
        contentLayers (list, optional): list of content layers from VGG16. Defaults to ["block3_conv3"].

    Returns:
        'Model' object: style loss Network.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    mdl = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=x0)
    # Compute style loss
    styleOutput = [mdl.get_layer(name).output for name in styleLayers]
    contentOutput = [mdl.get_layer(name).output for name in contentLayers]
    output = {"style": styleOutput, "content": contentOutput}
    return tf.keras.Model(inputs=x0, outputs=output)
