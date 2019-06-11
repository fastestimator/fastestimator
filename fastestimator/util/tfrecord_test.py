import os
from shutil import rmtree
from unittest import TestCase
import tensorflow as tf
from fastestimator.pipeline.static.preprocess import Reshape, Onehot, Minmax
from .tfrecord import TFRecorder


class TestTFRecorder(TestCase):
    def test__get_feature_info(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        tfr = TFRecorder({"x": x_train, "y": y_train}, ["x", "y"])
        tfr._get_feature_info()
        assert tfr.mb_per_csv_example == 0.000785
        assert sorted(tfr.feature_name_new) == sorted(['x', 'y'])
        assert tfr.feature_type_new == ['uint8', 'uint8']

    def test__preprocessing(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        tfr = TFRecorder({"x": x_train, "y": y_train}, ["x", "y"], [[Reshape([28, 28, 1]), Minmax()], [Onehot(10)]])
        tfr._get_feature_info()
        assert tfr.mb_per_csv_example == 1.6e-05
        assert sorted(tfr.feature_name_new) == sorted(['x', 'y'])
        assert tfr.feature_type_new == ['object', 'object']

    def test__prepare_training_dict(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        tfr = TFRecorder({"x": x_train, "y": y_train}, ["x", "y"])
        tfr._prepare_training_dict()
        assert tfr.num_train_example_csv == (len(x_train))

    def test_create_tfrecord(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        tfr = TFRecorder({"x": x_train, "y": y_train}, ["x", "y"], validation_data=0.2)
        tfr.create_tfrecord()
        assert os.path.isfile(os.path.join(tfr.save_dir, "summary0.json"))
        assert os.path.isfile(os.path.join(tfr.save_dir, "train0.tfrecord"))
        rmtree(tfr.save_dir)

