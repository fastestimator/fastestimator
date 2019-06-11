from fastestimator.pipeline.static.preprocess import Minmax, Reshape
from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.network.network import Network
from fastestimator.application.lenet import LeNet
import tensorflow as tf

def get_estimator(epochs=2, batch_size=32, optimizer="adam"):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    # Step 1: Define Pipeline
    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train= [[Reshape([28,28,1]), Minmax()], []])

    #Step2: Define Network
    network = Network(model=LeNet(input_name="x", output_name="y"),
                      loss="sparse_categorical_crossentropy",
                      metrics=["acc"],
                      optimizer=optimizer)
                      
    #Step3: Define Estimator
    estimator = Estimator(network= network,
                          pipeline=pipeline,
                          epochs= epochs)
    return estimator