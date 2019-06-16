from unittest import TestCase
from fastestimator.pipeline.static.filter import Filter
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape
import tensorflow as tf
import shutil

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
pipeline = Pipeline(feature_name=["x", "y"], train_data={'x': x_train, 'y': y_train},
                         validation_data={"x": x_test, "y": y_test}, batch_size=32,
                    transform_train=[[Reshape([28, 28, 1]), Minmax()],
                                     [Onehot(10)]])
filter = Filter('y', [1, 3], [0.2, 0.4])

class TestFilter(TestCase):
    def test_predicate_fn(self):
        pipeline._prepare()
        filenames = pipeline.file_names["train"]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda dataset: pipeline.read_and_decode(dataset))
        dataset.filter(lambda dataset: filter.predicate_fn(dataset))
        shutil.rmtree(pipeline.inputs)
