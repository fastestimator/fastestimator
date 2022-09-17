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


def unet3d_bn_before(input_size=(288, 288, 160, 1), output_classes=5, channels=32, batch_norm=True):
    inputs = Input(input_size)
    conv_config = {'padding': 'same', 'kernel_initializer': 'he_normal'}

    conv1 = Conv3D(channels, 3, **conv_config)(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(channels * 2, 3, **conv_config)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    if batch_norm:
        conv2 = BatchNormalization(axis=4)(pool1)
        conv2 = ReLU()(conv2)
    else:
        conv2 = ReLU()(pool1)
    conv2 = Conv3D(channels * 2, 3, **conv_config)(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(channels * 4, 3, **conv_config)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    if batch_norm:
        conv3 = BatchNormalization(axis=4)(pool2)
        conv3 = ReLU()(conv3)
    else:
        conv3 = ReLU()(pool2)
    conv3 = Conv3D(channels * 4, 3, **conv_config)(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(channels * 8, 3, **conv_config)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    if batch_norm:
        conv4 = BatchNormalization(axis=4)(pool3)
        conv4 = ReLU()(conv4)
    else:
        conv4 = ReLU()(pool3)
    conv4 = Conv3D(channels * 8, 3, **conv_config)(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(channels * 16, 3, **conv_config)(conv4)

    up5 = Conv3D(channels * 16, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=4)

    if batch_norm:
        conv5 = BatchNormalization(axis=4)(merge5)
        conv5 = ReLU()(conv5)
    else:
        conv5 = ReLU()(merge5)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(conv5)

    up6 = Conv3D(channels * 8, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=4)

    if batch_norm:
        conv6 = BatchNormalization(axis=4)(merge6)
        conv6 = ReLU()(conv6)
    else:
        conv6 = ReLU()(merge6)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(conv6)

    up7 = Conv3D(channels * 4, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=4)

    if batch_norm:
        conv7 = BatchNormalization(axis=4)(merge7)
        conv7 = ReLU()(conv7)
    else:
        conv7 = ReLU()(merge7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(conv7)
    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    return model
