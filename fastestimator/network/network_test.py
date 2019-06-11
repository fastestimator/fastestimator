from fastestimator.network.network import Network
from fastestimator.application.lenet import LeNet
import tensorflow as tf
import tempfile
import pytest
import os

def test_initialization1():
    tf.keras.backend.clear_session()
    network = Network(model=LeNet(input_name="x", output_name="y"),
                      loss = "binary_crossentrypy")
    assert os.path.exists(network.model_dir)

def test_initialization2():
    tf.keras.backend.clear_session()
    _ = Network(model=LeNet(input_name="x", output_name="y"),
                loss = "binary_crossentrypy", 
                model_dir= os.getcwd())
    assert True

def test_initialization3():
    tf.keras.backend.clear_session()
    network = Network(model=LeNet(input_name="x", output_name="y"),
                      loss = "binary_crossentrypy", 
                      model_dir= tempfile.mkdtemp())
    assert os.path.exists(network.model_dir)

def test_check_optimizer():
    tf.keras.backend.clear_session()
    with pytest.raises(ValueError):
        _ = Network(model=LeNet(input_name="x", output_name="y"),
                    loss = "binary_crossentrypy", 
                    optimizer=tf.train.AdamOptimizer())

def test_get_inout_list1():
    tf.keras.backend.clear_session()
    network = Network(model=LeNet(input_name="x", output_name="y"),
                      loss = "binary_crossentrypy")
    network._get_inout_list()
    assert network.input_names == ["x"] and network.output_names == ["y"]

def test_get_inout_list2():
    tf.keras.backend.clear_session()
    network = Network(model=LeNet(input_name="x_in", output_name="y_out"),
                      loss = "binary_crossentrypy")
    network._get_inout_list()
    assert network.input_names == ["x_in"] and network.output_names == ["y_out"]
