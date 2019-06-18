import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping_keras
from tensorflow.keras.callbacks import LearningRateScheduler as LearningRateScheduler_keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau_keras
from tensorflow.keras.callbacks import TensorBoard

from fastestimator.application.lenet import LeNet
from fastestimator.estimator.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.lrscheduler import CyclicScheduler
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape


class TestEstimator:
    tf.keras.backend.clear_session()
    g = tf.Graph()
    with g.as_default():
        pipeline = Pipeline(batch_size=16,
                            feature_name=["x", "y"],
                            train_data={"x": np.random.rand(500, 28, 28), "y":np.random.randint(0, 10, [500])},
                            validation_data={"x": np.random.rand(100, 28, 28), "y": np.random.randint(0, 10, [100])},
                            transform_train= [[Reshape([28,28,1]), Minmax()], [Onehot(10)]],
                            shuffle_buffer= 500)

        network = Network(model=LeNet(input_name="x", output_name="y"),
                        loss="categorical_crossentropy",
                        metrics=["acc"],
                        optimizer="adam")
        callbacks = [ReduceLROnPlateau(patience=2), 
                    EarlyStopping(patience=3),
                    ModelCheckpoint(os.path.join(network.model_dir, "BestModel.h5"), save_best_only=True),
                    LearningRateScheduler(schedule=CyclicScheduler()),
                    TensorBoard(log_dir=network.model_dir)]

        estimator = Estimator(network= network,
                            pipeline=pipeline,
                            epochs= 2,
                            callbacks=callbacks)

    def test_positive_case1(self):
        with self.g.as_default():
            self.estimator.fit()

    def test_positive_case2(self):
        with self.g.as_default():
            self.pipeline.num_examples["eval"] = 0
            self.estimator.callbacks = []
            self.pipeline.train_data=None
            self.pipeline.validation_data=None
            self.estimator.fit(self.pipeline.inputs)

    def test_negative_case1(self):
        with pytest.raises(ValueError):
            with self.g.as_default():
                self.estimator.fit()

    def test_negative_case2(self):
        with pytest.raises(ValueError):
            with self.g.as_default():
                self.estimator.callbacks = [EarlyStopping_keras(patience=2)]
                self.estimator.fit(self.pipeline.inputs)

    def test_negative_case3(self):
        with pytest.raises(ValueError):
            with self.g.as_default():
                self.estimator.callbacks = [ReduceLROnPlateau_keras(patience=2)]
                self.estimator.fit(self.pipeline.inputs)

    def test_negative_case4(self):
        with pytest.raises(ValueError):
            with self.g.as_default():
                self.estimator.callbacks = [LearningRateScheduler_keras(lambda epoch: 0.001*epoch)]
                self.estimator.fit(self.pipeline.inputs)

    def test_negative_case5(self):
        with pytest.raises(ValueError):
            with self.g.as_default():
                self.pipeline.feature_name = ["x/", "y/"]
                self.estimator.fit(self.pipeline.inputs)
