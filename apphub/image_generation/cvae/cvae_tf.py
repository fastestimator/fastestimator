import pdb
import tempfile
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.dataset.data.mnist import load_data
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ExpandDims, Minmax
from fastestimator.op.tensorop.model import ModelOp

LATENT_DIM = 50

def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * tf.constant(math.pi))
    return tf.reduce_sum(-.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

class Binarize(NumpyOp):
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
        state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        pdb.set_trace()
        data = (data > 0.5).astype(np.float32)
        return data

class SplitOp(TensorOp):
    """To split the infer net output into two """
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
        state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        mean, logvar = tf.split(data, num_or_size_splits=2, axis=1)
        return mean, logvar

class ReparameterizepOp(TensorOp):
    """Reparameterization trick. Ensures grads pass thru the sample to the infer net parameters"""
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
        state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        mean, logvar = data
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

class CVAELoss(TensorOp):
    """Convolutional variational auto-endcoder loss"""
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
        state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        x, mean, logvar, z, x_logit = data
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = _log_normal_pdf(z, 0., 0.)
        logqz_x = _log_normal_pdf(z, mean, logvar)
        return -(logpx_z + logpz - logqz_x)

def encoder_net():
    infer_model = tf.keras.Sequential()
    infer_model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    infer_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Flatten())
    infer_model.add(tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM))
    return infer_model


def decoder_net():
    generative_model = tf.keras.Sequential()
    generative_model.add(tf.keras.layers.InputLayer(input_shape=(LATENT_DIM, )))
    generative_model.add(tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu))
    generative_model.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME"))
    return generative_model



def get_estimator(batch_size=256, epochs=50, steps_per_epoch=None, model_dir=tempfile.mkdtemp()):
    train_data, test_data = load_data()

    pipeline = fe.Pipeline(train_data=train_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[
                                ExpandDims(inputs="x", outputs="x"),
                                Minmax(inputs="x", outputs="x"),
                                Binarize(inputs="x", outputs="x")
                           ])

    encode_model = fe.build(model_fn=encoder_net, optimizer_fn="adam")
    decode_model = fe.build(model_fn=decoder_net, optimizer_fn="adam")

    network = fe.Network(ops=[
        ModelOp(model=encode_model, inputs="x"),
        SplitOp(inputs="x", outputs=("mean", "logvar")),
        ReparameterizepOp(inputs=("means", "logvar"), outputs="z"),
        ModelOp(model=decode_model, inputs="z", outputs="x_logit"),
        CVAELoss(inputs=("x", "mean", "logvar", "z", "x_logit"), outputs="loss")
    ])

    estimator = fe.Estimator(
        pipeline = pipeline,
        network = network,
        epochs = epochs
    )

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
