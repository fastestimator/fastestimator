from fastestimator.pipeline.static.preprocess import Minmax
from fastestimator.pipeline.pipeline import Pipeline
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1)

pipeline = Pipeline(batch_size=32,
                    feature_name=["x", "y"],
                    train_data={"x": x_train, "y": y_train},
                    validation_data= {"x": x_eval, "y": y_eval},
                    transform_train= [[Minmax()], []])

pipeline.benchmark(num_steps=3000, mode="train")