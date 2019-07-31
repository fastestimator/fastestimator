import numpy as np
import tensorflow as tf
from fastestimator.architecture.lenet import LeNet
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Accuracy, Recall
from fastestimator.network.loss import SparseCategoricalCrossentropy
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.preprocess import Minmax

def get_estimator(epochs=2, batch_size=32):
    #step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    data = {"train":  {"x": np.expand_dims(x_train, -1),  "y": y_train},  "eval": {"x": np.expand_dims(x_eval, -1),  "y": y_eval}}
    pipeline = Pipeline(batch_size=batch_size,
                        data=data,
                        ops=Minmax(inputs="x", outputs="x"))

    #step 2. prepare model
    model = build(keras_model=LeNet(), loss=SparseCategoricalCrossentropy(true_key="y", pred_key="y_pred"), optimizer="adam")
    network = Network(ops=ModelOp(inputs="x", model=model, outputs="y_pred"))

    #step 3.prepare estimator
    estimator = Estimator(network=network,
                          pipeline=pipeline,
                          epochs=epochs,
                          traces=[Accuracy(true_key="y", pred_key="y_pred"), Recall(true_key="y", pred_key="y_pred")])
    return estimator