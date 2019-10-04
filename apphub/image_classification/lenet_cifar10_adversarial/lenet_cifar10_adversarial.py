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
import tempfile

import tensorflow as tf

from fastestimator import Estimator, Network, Pipeline, build
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop import AdversarialSample, Average, Minmax, ModelOp, SparseCategoricalCrossentropy
from fastestimator.schedule import Scheduler
from fastestimator.trace import Accuracy, ConfusionMatrix, ModelSaver


def get_estimator(epochs=10, batch_size=32, epsilon=0.01, warmup=0, steps_per_epoch=None, model_dir=tempfile.mkdtemp()):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    num_classes = 10

    pipeline = Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))

    model = build(model_def=lambda: LeNet(input_shape=x_train.shape[1:], classes=num_classes),
                  model_name="LeNet",
                  optimizer="adam",
                  loss_name="loss")

    adv_img = {warmup: AdversarialSample(inputs=("loss", "x"), outputs="x_adverse", epsilon=epsilon, mode="train")}
    adv_eval = {warmup: ModelOp(inputs="x_adverse", model=model, outputs="y_pred_adverse", mode="train")}
    adv_loss = {
        warmup: SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred_adverse", outputs="adverse_loss", mode="train")
    }
    adv_avg = {warmup: Average(inputs=("loss", "adverse_loss"), outputs="loss", mode="train")}

    network = Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred", track_input=True),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss"),
        Scheduler(adv_img),
        Scheduler(adv_eval),
        Scheduler(adv_loss),
        Scheduler(adv_avg)
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        ConfusionMatrix(true_key="y", pred_key="y_pred", num_classes=num_classes),
        ModelSaver(model_name="LeNet", save_dir=model_dir, save_freq=2)
    ]

    estimator = Estimator(network=network,
                          pipeline=pipeline,
                          epochs=epochs,
                          traces=traces,
                          steps_per_epoch=steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
