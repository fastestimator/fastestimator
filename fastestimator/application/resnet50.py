from tensorflow.keras.applications.resnet50 import ResNet50 as ResNet50_Keras
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

def ResNet50(input_name, output_name, input_shape, classes=1000, weights=None):
    inputs = Input(input_shape, name=input_name)
    x = ResNet50_Keras(weights=weights, input_shape=input_shape, include_top=False, pooling='avg')(inputs)
    outputs = Dense(classes, activation='softmax', name=output_name)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model