import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3 as InceptionV3_keras

def InceptionV3(input_name, output_name, input_shape, classes=1000, weights=None):
    inputs = Input(input_shape, name=input_name)
    x = InceptionV3_keras(weights=weights, input_shape=input_shape, include_top=False, pooling='avg')(inputs)
    outputs = Dense(classes, activation='softmax', name=output_name)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model