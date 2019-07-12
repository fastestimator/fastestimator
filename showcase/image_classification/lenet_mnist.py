import numpy as np
import tensorflow as tf

from fastestimator.architecture.lenet import LeNet
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.network import Network, prepare_model
from fastestimator.estimator.trace import Accuracy, ConfusionMatrix
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax
from fastestimator.network.loss import SparseCategoricalCrossentropy
from fastestimator.network.operation import Operation


def get_estimator(epochs=2, batch_size=32, optimizer="adam"):
    #prepare the data
    # (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    # x_train = np.expand_dims(x_train, -1)
    # x_eval = np.expand_dims(x_eval, -1)

    #define pipeline for training
    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        # train_data={"x": x_train, "y": y_train},
                        # validation_data={"x": x_eval, "y": y_eval},
                        transform_train=[[Minmax()], []])

    lenet = prepare_model(keras_model=LeNet(), loss=SparseCategoricalCrossentropy(true_key="y", pred_key="y_pred"), optimizer=tf.optimizers.Adam())
    
    #define network
    network = Network(ops=Operation(key_in= "x", link=lenet, key_out="y_pred"))

    traces = [Accuracy(true_key="y", pred_key="y_pred"), ConfusionMatrix(true_key="y", pred_key="y_pred", num_classes=10)]

    #define estimator
    estimator = Estimator(network=network,
                          pipeline=pipeline,
                          epochs=epochs,
                          traces=traces)
    return estimator
