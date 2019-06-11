import sys
from unittest import TestCase
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import testing_utils
import numpy as np

from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Reshape, Onehot
from fastestimator.network.network import Network
from fastestimator.network.lrscheduler import CyclicScheduler
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from io import StringIO
from contextlib import contextmanager

TRAIN_SAMPLES = 1000
TEST_SAMPLES = 100
NUM_CLASSES = 2
INPUT_DIM = 3
BATCH_SIZE = 10

def _build_model(input_name, output_name, input_shape=(INPUT_DIM,INPUT_DIM,1)):
    x0 = tf.keras.layers.Input(input_shape, name=input_name)
    x = tf.keras.layers.Flatten()(x0)
    x = tf.keras.layers.Dense(3, activation="relu")(x)
    x = tf.keras.layers.Dense(2, activation="softmax", name=output_name)(x)
    return tf.keras.Model(inputs=x0, outputs=x)

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class FastEstimatorCallbacksTest(TestCase):
    K.clear_session()
    np.random.seed(1337)
    tf.random.set_random_seed(1337)

    g = tf.Graph()


    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
              train_samples=TRAIN_SAMPLES,
              test_samples=TEST_SAMPLES,
              input_shape=(INPUT_DIM, INPUT_DIM,1),
              num_classes=NUM_CLASSES)


    with g.as_default():
        pipeline = Pipeline(feature_name=["x", "y"],
                            train_data={"x": x_train, "y": y_train},
                            validation_data={"x": x_test, "y": y_test},
                            batch_size=BATCH_SIZE,
                            transform_train=[
                                [Reshape([INPUT_DIM, INPUT_DIM, 1])],
                                [Onehot(NUM_CLASSES)]
                            ]
                    )
        network = Network(model=_build_model(input_name="x", output_name="y"),
                        loss="categorical_crossentropy",
                        metrics=["acc"],
                        optimizer="adam")


        estimator = Estimator(network=network,
                            pipeline=pipeline,
                            epochs=5)



    def test_LearningRateScheduler_case1(self):
        # testing linear decay
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            schedule = CyclicScheduler(num_cycle=1, cycle_multiplier=1, decrease_method="linear")
            callbacks = [LearningRateScheduler(schedule)]
            self.estimator.callbacks = callbacks
            self.estimator.fit()

        def _compute_expected_lr():
            lr_ratio_start = 1.0
            lr_ratio_end = 1e-6
            step_start = 0
            step_end = 5 * (TRAIN_SAMPLES / BATCH_SIZE)
            global_steps = step_end - 1
            
            slope = (lr_ratio_start - lr_ratio_end)/(step_start - step_end)
            intercept = lr_ratio_start - slope * step_start
            lr_ratio = slope * global_steps + intercept
            return lr_ratio

        expected_lr_ratio = _compute_expected_lr()
        expected_lr = 1e-3 * expected_lr_ratio
        self.assertAlmostEqual(K.get_value(self.network.optimizer.lr), expected_lr)
        K.clear_session()

    def test_LearningRateScheduler_case2(self):
        # testing linear decay
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            schedule = CyclicScheduler(num_cycle=1, cycle_multiplier=1, decrease_method="cosine")
            callbacks = [LearningRateScheduler(schedule)]
            self.estimator.callbacks = callbacks
            self.estimator.fit()

        expected_lr = 1e-4 * 1e-6
        self.assertAlmostEqual(K.get_value(self.network.optimizer.lr), expected_lr)
        K.clear_session()

    def test_ReduceLROnPlateau_case1(self):
        # This is the case where it should reduce lr 
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            callbacks = [ReduceLROnPlateau(patience=1, factor=0.1, min_delta=10, cooldown=5)]
            self.estimator.callbacks = callbacks
            self.estimator.fit()

        self.assertAlmostEqual(K.get_value(self.network.optimizer.lr), 1e-4)
        K.clear_session()

    def test_ReduceLROnPlateau_case2(self):
        # This is the case where it should not reduce lr 
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            callbacks = [ReduceLROnPlateau(patience=1, factor=0.1, min_delta=0, cooldown=5)]
            self.estimator.callbacks = callbacks
            self.estimator.fit()

        self.assertAlmostEqual(K.get_value(self.network.optimizer.lr), 1e-3)
        K.clear_session()

    def test_EarlyStopping_case1(self):
        # this is the case where it should early stop
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            callbacks = [EarlyStopping(patience=0, min_delta=1)]
            self.estimator.callbacks = callbacks

            with captured_output() as (out, _):
                self.estimator.fit()
                output = out.getvalue().strip()

        self.assertIn("EarlyStopping", output)
        K.clear_session()

    def test_EarlyStopping_case2(self):
        # this is the case where it should not early stop
        with self.g.as_default():
            K.get_session().run(tf.initializers.global_variables())
            callbacks = [EarlyStopping(patience=10)]
            self.estimator.callbacks = callbacks

            with captured_output() as (out, _):
                self.estimator.fit()
                output = out.getvalue().strip()

        self.assertNotIn("EarlyStopping", output)
        K.clear_session()

