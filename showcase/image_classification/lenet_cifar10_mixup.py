import tensorflow as tf
import tensorflow_probability as tfp

from fastestimator.architecture.lenet import LeNet
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Accuracy, ConfusionMatrix
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.preprocess import Minmax


class Network:
    def __init__(self, shape, num_classes, alpha=1.0):
        self.model = LeNet(input_shape=shape, classes=num_classes)
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.SparseCategoricalCrossentropy()
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

        # Could do random mix-up using tf.gather() on a shuffled index list, but batches are already randomly ordered,
        # so just need to roll by 1 to get a random combination of inputs
        x_mix = lam * batch['x'] + (1 - lam) * tf.roll(batch['x'], shift=1, axis=0)
        y_mix = tf.roll(batch['y'], shift=1, axis=0)

        with tf.GradientTape() as tape:
            predictions = self.model(x_mix)
            loss = lam * self.loss(batch["y"], predictions) + (1 - lam) * self.loss(y_mix, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions, loss

    def eval_op(self, batch):
        predictions = self.model(batch["x"], training=False)
        loss = self.loss(batch["y"], predictions)
        return predictions, loss


def get_estimator(epochs=2, batch_size=32, alpha=1.0):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    num_classes = 10

    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train=[[Minmax()], []])

    traces = [Accuracy(y_true_key="y"), ConfusionMatrix(y_true_key="y", num_classes=num_classes)]

    estimator = Estimator(
        network=Network(shape=x_train.shape[1:], num_classes=num_classes, alpha=alpha),
        pipeline=pipeline,
        epochs=epochs,
        traces=traces)
    return estimator
