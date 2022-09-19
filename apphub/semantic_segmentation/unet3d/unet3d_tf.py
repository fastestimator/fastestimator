from tensorflow import sigmoid
from tensorflow.keras.layers import BatchNormalization, Conv3D, Input, MaxPooling3D, ReLU, UpSampling3D, concatenate
from tensorflow.keras.models import Model


def unet3d(input_size=(288, 288, 160, 1), output_classes=5, channels=32, batch_norm=True):
    inputs = Input(input_size)
    conv_config = {'padding': 'same', 'kernel_initializer': 'he_normal'}

    conv1 = Conv3D(channels, 3, **conv_config)(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(channels * 2, 3, **conv_config)(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(channels * 2, 3, **conv_config)(pool1)
    if batch_norm:
        conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(channels * 4, 3, **conv_config)(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(channels * 4, 3, **conv_config)(pool2)
    if batch_norm:
        conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(channels * 8, 3, **conv_config)(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(channels * 8, 3, **conv_config)(pool3)
    if batch_norm:
        conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(channels * 16, 3, **conv_config)(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)

    up5 = Conv3D(channels * 16, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=4)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(merge5)
    if batch_norm:
        conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)

    up6 = Conv3D(channels * 8, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=4)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(merge6)
    if batch_norm:
        conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv3D(channels * 4, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=4)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(merge7)
    if batch_norm:
        conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)
    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    return model


def unet3d_3plus(input_size=(288, 288, 160, 1), output_classes=5, channels=32):
    inputs = Input(input_size)
    conv_config = {'activation': None, 'padding': 'same', 'kernel_initializer': 'he_normal'}

    conv1 = Conv3D(channels, 3, **conv_config)(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(channels * 2, 3, **conv_config)(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(channels * 2, 3, **conv_config)(pool1)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(channels * 4, 3, **conv_config)(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(channels * 4, 3, **conv_config)(pool2)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(channels * 8, 3, **conv_config)(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(channels * 8, 3, **conv_config)(pool3)
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(channels * 16, 3, **conv_config)(conv4)
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)

    up5_4 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv4))
    up5_4 = BatchNormalization(axis=4)(up5_4)
    up5_4 = ReLU()(up5_4)
    up5_3 = Conv3D(channels, 2, **conv_config)(conv3)
    up5_3 = BatchNormalization(axis=4)(up5_3)
    up5_3 = ReLU()(up5_3)
    down5_2 = Conv3D(channels, 2, **conv_config)(MaxPooling3D(pool_size=(2, 2, 2))(conv2))
    down5_2 = BatchNormalization(axis=4)(down5_2)
    down5_2 = ReLU()(down5_2)
    down5_1 = Conv3D(channels, 2, **conv_config)(MaxPooling3D(pool_size=(4, 4, 4))(conv1))
    down5_1 = BatchNormalization(axis=4)(down5_1)
    down5_1 = ReLU()(down5_1)
    merge5 = concatenate([up5_4, up5_3, down5_2, down5_1], axis=4)
    merge5 = BatchNormalization(axis=4)(merge5)
    merge5 = ReLU()(merge5)
    conv5 = Conv3D(channels * 2, 3, **conv_config)(merge5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)

    up6_4 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(4, 4, 4))(conv4))
    up6_4 = BatchNormalization(axis=4)(up6_4)
    up6_4 = ReLU()(up6_4)
    up6_3 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv5))
    up6_3 = BatchNormalization(axis=4)(up6_3)
    up6_3 = ReLU()(up6_3)
    up6_2 = Conv3D(channels, 2, **conv_config)(conv2)
    up6_2 = BatchNormalization(axis=4)(up6_2)
    up6_2 = ReLU()(up6_2)
    down6_1 = Conv3D(channels, 2, **conv_config)(MaxPooling3D(pool_size=(2, 2, 2))(conv1))
    down6_1 = BatchNormalization(axis=4)(down6_1)
    down6_1 = ReLU()(down6_1)
    merge6 = concatenate([up6_4, up6_3, up6_2, down6_1], axis=4)
    merge6 = BatchNormalization(axis=4)(merge6)
    merge6 = ReLU()(merge6)
    conv6 = Conv3D(channels * 2, 3, **conv_config)(merge6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)

    up7_4 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(8, 8, 8))(conv4))
    up7_4 = BatchNormalization(axis=4)(up7_4)
    up7_4 = ReLU()(up7_4)
    up7_3 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(4, 4, 4))(conv5))
    up7_3 = BatchNormalization(axis=4)(up7_3)
    up7_3 = ReLU()(up7_3)
    up7_2 = Conv3D(channels, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv6))
    up7_2 = BatchNormalization(axis=4)(up7_2)
    up7_2 = ReLU()(up7_2)
    conv7_1 = Conv3D(channels, 2, **conv_config)(conv1)
    conv7_1 = BatchNormalization(axis=4)(conv7_1)
    conv7_1 = ReLU()(conv7_1)
    merge7 = concatenate([up7_4, up7_3, up7_2, conv7_1], axis=4)
    merge7 = BatchNormalization(axis=4)(merge7)
    merge7 = ReLU()(merge7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(merge7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)

    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    return model


def attentionUnet3D(input_size=(288, 288, 160, 1), output_classes=5, channels=32, batch_norm=True):
    inputs = Input(input_size)
    conv_config = {'padding': 'same', 'kernel_initializer': 'he_normal'}

    conv1 = Conv3D(channels, 3, **conv_config)(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(channels * 2, 3, **conv_config)(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(channels * 2, 3, **conv_config)(pool1)
    if batch_norm:
        conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(channels * 4, 3, **conv_config)(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(channels * 4, 3, **conv_config)(pool2)
    if batch_norm:
        conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(channels * 8, 3, **conv_config)(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(channels * 8, 3, **conv_config)(pool3)
    if batch_norm:
        conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(channels * 16, 3, **conv_config)(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)

    up5 = Conv3D(channels * 16, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv4))
    conv3 = attention_block(channels * 16, decoder_input=up5, encoder_input=conv3)
    merge5 = concatenate([conv3, up5], axis=4)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(merge5)
    if batch_norm:
        conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)

    up6 = Conv3D(channels * 8, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv5))
    conv2 = attention_block(channels * 8, decoder_input=up6, encoder_input=conv2)
    merge6 = concatenate([conv2, up6], axis=4)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(merge6)
    if batch_norm:
        conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv3D(channels * 4, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv6))
    conv1 = attention_block(channels * 4, decoder_input=up7, encoder_input=conv1)
    merge7 = concatenate([conv1, up7], axis=4)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(merge7)
    if batch_norm:
        conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)

    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    return model


def attention_block(n_filters: int, decoder_input, encoder_input):
    """An attention unit for Attention Unet.

    Args:
        n_filters: How many filters for the convolution layer.
        decoder_input: Input tensor in the decoder section.
        encoder_input: Input tensor in the encoder section.

    Return:
        Output Keras tensor.
    """
    c1 = Conv3D(n_filters, kernel_size=1)(decoder_input)
    c1 = BatchNormalization()(c1)
    x1 = Conv3D(n_filters, kernel_size=1)(encoder_input)
    x1 = BatchNormalization()(x1)
    att = ReLU()(x1 + c1)
    att = Conv3D(1, kernel_size=1)(att)
    att = BatchNormalization()(att)
    att = sigmoid(att)
    return encoder_input * att
