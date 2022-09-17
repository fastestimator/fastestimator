from tensorflow.keras.layers import Conv3D, Dropout, Input, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model


def unet3d(input_size=(288, 288, 160, 1), output_classes=5, channels=32, batch_norm=True, bn_before=False):
    inputs = Input(input_size)
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}

    conv1 = Conv3D(channels, 3, **conv_config)(inputs)
    conv1 = Conv3D(channels * 2, 3, **conv_config)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(channels * 2, 3, **conv_config)(pool1)
    conv2 = Conv3D(channels * 4, 3, **conv_config)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(channels * 4, 3, **conv_config)(pool2)
    conv3 = Conv3D(channels * 8, 3, **conv_config)(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Conv3D(channels * 8, 3, **conv_config)(pool3)
    conv4 = Conv3D(channels * 16, 3, **conv_config)(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv3D(channels * 16, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=4)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(merge5)
    conv5 = Conv3D(channels * 8, 3, **conv_config)(conv5)

    up6 = Conv3D(channels * 8, 2, **conv_config)(UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=4)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(merge6)
    conv6 = Conv3D(channels * 4, 3, **conv_config)(conv6)

    up7 = Conv3D(channels * 4, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=4)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(merge7)
    conv7 = Conv3D(channels * 2, 3, **conv_config)(conv7)
    conv8 = Conv3D(output_classes, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    return model
