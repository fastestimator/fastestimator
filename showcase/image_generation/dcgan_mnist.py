from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from tensorflow.keras import layers
from fastestimator.network.loss import Loss
from fastestimator.network.network import Network, prepare_model
import tensorflow as tf
from fastestimator.network.operation import Operation
import numpy as np

class g_loss(Loss):
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def calculate_loss(self, batch, prediction):
        return self.cross_entropy(tf.ones_like(prediction["y_pred_fake"]), prediction["y_pred_fake"])

class d_loss(Loss):
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def calculate_loss(self, batch, prediction):
        real_loss = self.cross_entropy(tf.ones_like(prediction["y_pred_true"]), prediction["y_pred_true"])
        fake_loss = self.cross_entropy(tf.zeros_like(prediction["y_pred_fake"]), prediction["y_pred_fake"])
        total_loss = real_loss + fake_loss
        return total_loss

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

class Myrescale:
    def transform(self, data, decoded_data=None):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data

class myOperation(Operation):
    def forward(self, batch, prediction, mode, epoch):
        data = tf.random.normal([32, 100])
        for block in self.link:
            if isinstance(block, tf.keras.Model):
                data = block(data, training=mode=="train")
            elif block.mode in [mode, "both"]:
                data = block(data)
        prediction[self.key_out] = data
        return prediction

def get_estimator():
    (x_train, _), (x_eval, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    pipeline = Pipeline(batch_size=32,
                        feature_name=["x"],
                        train_data={"x": x_train},
                        validation_data={"x": x_eval},
                        transform_train= [[Myrescale()], []])

    g = prepare_model(keras_model=make_generator_model(), loss=g_loss(), optimizer=tf.optimizers.Adam(1e-4))
    d = prepare_model(keras_model=make_discriminator_model(), loss=d_loss(), optimizer=tf.optimizers.Adam(1e-4))
    network = Network(ops=[myOperation(key_in= "noise", link=[g, d], key_out="y_pred_fake"),
                           Operation(key_in= "x", link=d, key_out="y_pred_true")])

    estimator = Estimator(network= network,
                          pipeline=pipeline,
                          epochs= 2)
    return estimator