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
import numpy as np
import tensorflow as tf
import fastestimator as fe

from tensorflow.keras import layers, losses
from tensorflow.keras import Sequential

from fastestimator.op import TensorOp
from fastestimator.op.tensorop import ModelOp, UpdateOp, Gradients
from fastestimator.op.tensorop.loss import MeanSquaredError, Loss
from fastestimator.trace import ModelSaver


class MetaModelOp(ModelOp):
    def _single_forward(self, data):
        return self.model(data, training=True)

    def forward(self, data, state):
        out = tf.map_fn(fn=self._single_forward, elems=data, dtype=tf.float32)
        return out


class MetaMSE(MeanSquaredError):
    def forward(self, data, state):
        true, pred = data
        out = self.loss_obj(true, pred)
        return tf.reduce_mean(out, axis=1)


class InnerGradientOp(TensorOp):
    def __init__(self, loss, model, outputs):
        super().__init__(inputs=loss, outputs=outputs, mode="train")
        self.model = model

    def forward(self, data, state):
        loss = data
        tape = state['tape']
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients, self.model.trainable_variables


class InnerUpdateOp(TensorOp):
    def __init__(self, inputs, outputs, inner_lr):
        super().__init__(inputs=inputs, outputs=outputs, mode="train")
        self.inner_lr = inner_lr

    def forward(self, data, state):
        g, v = data
        return [v_ - self.inner_lr * g_ for g_, v_ in zip(g, v)]


class MetaForwardOp(TensorOp):
    def forward(self, data, state):
        x0, model_var = data

        def _single_forward(x):
            out = tf.nn.relu(tf.matmul(x, model_var[0]) + model_var[1])
            for i in range(2, len(model_var) - 2, 2):
                out = tf.nn.relu(tf.matmul(out, model_var[2]) + model_var[3])
            out = tf.matmul(out, model_var[-2]) + model_var[-1]
            return out

        return tf.map_fn(_single_forward, elems=x0, dtype=tf.float32)


def generate_random_sine(amp_range=[0.1, 5.0], phase_range=[0, np.pi], x_range=[-5.0, 5.0], K=10):
    while True:
        a = np.random.uniform(amp_range[0], amp_range[1])
        b = np.random.uniform(phase_range[0], phase_range[1])
        x = np.random.uniform(x_range[0], x_range[1], 2 * K).astype(np.float32)
        y = a * np.sin(x + b).astype(np.float32)
        yield {
            "x_meta_train": np.expand_dims(x[:K], axis=-1),
            "x_meta_test": np.expand_dims(x[K:], axis=-1),
            "y_meta_train": np.expand_dims(y[:K], axis=-1),
            "y_meta_test": np.expand_dims(y[K:], axis=-1),
            "amp": a,
            "phase": b
        }


def build_sine_model():
    mdl = Sequential()
    mdl.add(layers.Dense(40, input_shape=(1, ), activation="relu"))
    mdl.add(layers.Dense(40, activation="relu"))
    mdl.add(layers.Dense(1))
    return mdl


def get_estimator(batch_size=25, epochs=1, steps_per_epoch=20000, validation_steps=None, model_dir=tempfile.mkdtemp()):
    pipeline = fe.Pipeline(data={"train": generate_random_sine}, batch_size=batch_size)

    meta_model = fe.build(model_def=build_sine_model, model_name="meta_model", loss_name="meta_loss", optimizer="adam")

    network = fe.Network(ops=[
        MetaModelOp(inputs="x_meta_train", outputs="y_meta_pred", model=meta_model),
        MetaMSE(inputs=("y_meta_train", "y_meta_pred"), outputs="inner_loss"),
        InnerGradientOp(loss="inner_loss", model=meta_model, outputs=("inner_grad", "model_var")),
        InnerUpdateOp(inputs=("inner_grad", "model_var"), outputs="model_var", inner_lr=1e-3),
        MetaForwardOp(inputs=("x_meta_test", "model_var"), outputs="y_pred"),
        MetaMSE(inputs=("y_meta_test", "y_pred"), outputs="meta_loss"),
        Gradients(loss="meta_loss", models=meta_model, outputs="meta_grad"),
        UpdateOp(model=meta_model, gradients="meta_grad")
    ])

    traces = [ModelSaver(model_name="meta_model", save_dir=model_dir, save_best=False)]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             traces=traces,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
