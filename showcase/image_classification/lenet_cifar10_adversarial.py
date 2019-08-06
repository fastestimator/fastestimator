# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import tensorflow_probability as tfp

from fastestimator.architecture.lenet import LeNet
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Accuracy, ConfusionMatrix
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax


class Network:
    def __init__(self, shape, num_classes, alpha=1.0, epsilon=0.1):
        self.model = LeNet(input_shape=shape, classes=num_classes)
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.epsilon = tf.constant(epsilon)
        self.alpha = tf.constant(alpha)
        self.beta = tfp.distributions.Beta(alpha, alpha)

    def train_op(self, batch):
        if self.alpha <= 0:
            with tf.GradientTape() as tape:
                predictions = self.model(batch["x"])
                loss = self.loss(batch["y"], predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return predictions, loss

        lam = self.beta.sample()
        x_clean = batch['x']

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_clean)
            predictions_clean = self.model(x_clean)
            loss_clean = self.loss(batch["y"], predictions_clean)
            with tape.stop_recording():
                grad_clean = tape.gradient(loss_clean, x_clean)
                x_dirty = tf.clip_by_value(x_clean + self.epsilon * tf.sign(grad_clean), tf.reduce_min(x_clean),
                                           tf.reduce_max(x_clean))
            predictions_dirty = self.model(x_dirty)
            loss = lam * loss_clean + (1 - lam) * self.loss(batch["y"], predictions_dirty)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions_clean, loss

    def eval_op(self, batch):
        predictions = self.model(batch["x"], training=False)
        loss = self.loss(batch["y"], predictions)
        return predictions, loss


def get_estimator(epochs=2, batch_size=32, alpha=1.0, epsilon=0.1):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    num_classes = 10

    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train=[[Minmax()], []])

    traces = [Accuracy(true_key="y", pred_key="y_pred"), ConfusionMatrix(y_true_key="y", num_classes=num_classes)]

    estimator = Estimator(
        network=Network(shape=x_train.shape[1:], num_classes=num_classes, alpha=alpha, epsilon=epsilon),
        pipeline=pipeline,
        epochs=epochs,
        traces=traces)
    return estimator
