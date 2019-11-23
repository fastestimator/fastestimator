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
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model, Sequential, layers
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2

import fastestimator as fe
from fastestimator.dataset.omniglot import get_batch, load_data, load_eval_data, one_shot_trial
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Augmentation2D, BinaryCrossentropy, Minmax, ModelOp
from fastestimator.schedule import LRSchedule
from fastestimator.trace import Accuracy, EarlyStopping, LRController, ModelSaver, Trace


class ConditionalAugmentation(TensorOp):
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=1.,
                 augment_prob=0.5):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.augment_prob = augment_prob
        self.aug2d = Augmentation2D(inputs=inputs,
                                    outputs=outputs,
                                    mode=mode,
                                    rotation_range=rotation_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range)

    def forward(self, data, state):
        random_number = tf.random.uniform([])
        if random_number < self.augment_prob:
            data = self.aug2d.forward(data, state)
        return data


class LRDecaySchedule(LRSchedule):
    def __init__(self, decay_rate=0.99):
        super().__init__(schedule_mode="epoch")
        self.decay_rate = decay_rate

    def schedule_fn(self, current_step_or_epoch, lr):
        if current_step_or_epoch > 0:
            lr = np.float32(lr * self.decay_rate)
        return lr


class OneShotAccuracy(Trace):
    def __init__(self, model, img_list, N=20, trials_per_alphabet=40, mode="eval", output_name="one_shot_accuracy"):

        super().__init__(outputs=output_name, mode=mode)
        self.model = model
        self.total = 0
        self.correct = 0
        self.output_name = output_name
        self.img_list = img_list
        self.N = N
        self.trials_per_alphabet = trials_per_alphabet

    def on_epoch_begin(self, state):
        self.total = 0
        self.correct = 0

    def on_epoch_end(self, state):
        for alphabet in range(len(self.img_list)):

            for _ in range(self.trials_per_alphabet):
                input_img = one_shot_trial(self.img_list[alphabet], self.N)
                prediction_score = self.model(input_img).numpy()

                if np.argmax(prediction_score) == 0 and prediction_score.std() > 0.01:
                    self.correct += 1

                self.total += 1

        state[self.output_name] = self.correct / self.total


def siamese_network(input_shape=(105, 105, 1), classes=1):
    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    #Creating the convnet which shares weights between the left and right legs of Siamese network
    siamese_convnet = Sequential()

    siamese_convnet.add(
        layers.Conv2D(filters=64,
                      kernel_size=10,
                      strides=1,
                      input_shape=input_shape,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=128,
                      kernel_size=7,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=128,
                      kernel_size=4,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.MaxPooling2D(pool_size=(2, 2)))

    siamese_convnet.add(
        layers.Conv2D(filters=256,
                      kernel_size=4,
                      strides=1,
                      activation='relu',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                      kernel_regularizer=l2(1e-2),
                      bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    siamese_convnet.add(layers.Flatten())

    siamese_convnet.add(
        layers.Dense(4096,
                     activation='sigmoid',
                     kernel_initializer=RandomNormal(mean=0, stddev=0.2),
                     kernel_regularizer=l2(1e-4),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01)))

    encoded_left_input = siamese_convnet(left_input)
    encoded_right_input = siamese_convnet(right_input)

    l1_encoded = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_left_input, encoded_right_input])

    output = layers.Dense(classes,
                          activation='sigmoid',
                          kernel_initializer=RandomNormal(mean=0, stddev=0.2),
                          bias_initializer=RandomNormal(mean=0.5, stddev=0.01))(l1_encoded)

    return Model(inputs=[left_input, right_input], outputs=output)


def get_estimator(epochs=200, batch_size=128, steps_per_epoch=500, validation_steps=100, model_dir=tempfile.mkdtemp()):
    # step 1. prepare pipeline
    train_path, eval_path = load_data()
    data = {
        "train": lambda: get_batch(train_path, batch_size=batch_size),
        "eval": lambda: get_batch(eval_path, batch_size=batch_size, is_train=False)
    }
    pipeline = fe.Pipeline(
        data=data,
        batch_size=batch_size,
        ops=[
            ConditionalAugmentation(inputs="x_a",
                                    outputs="x_a",
                                    mode="train",
                                    rotation_range=[-10, 10],
                                    shear_range=[-0.3 * 180 / np.pi, 0.3 * 180 / np.pi],
                                    zoom_range=[0.8, 1.2],
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    augment_prob=0.89),
            ConditionalAugmentation(inputs="x_b",
                                    outputs="x_b",
                                    mode="train",
                                    rotation_range=[-10, 10],
                                    shear_range=[-0.3 * 180 / np.pi, 0.3 * 180 / np.pi],
                                    zoom_range=[0.8, 1.2],
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    augment_prob=0.89),
            Minmax(inputs="x_a", outputs="x_a"),
            Minmax(inputs="x_b", outputs="x_b")
        ])

    # step 2. prepare model
    model = fe.build(model_def=siamese_network,
                     model_name="siamese_net",
                     optimizer=Adam(learning_rate=1e-4),
                     loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs=["x_a", "x_b"], model=model, outputs="y_pred"),
        BinaryCrossentropy(inputs=("y", "y_pred"), outputs="loss")
    ])

    # Defining learning rate schedule
    lr_scheduler = LRDecaySchedule(decay_rate=0.99)

    # Loading images for validation one-shot accuracy
    val_img_list = load_eval_data(path=eval_path, is_test=False)

    # step 3.prepare estimator
    traces = [
        LRController(model_name="siamese_net", lr_schedule=lr_scheduler),
        OneShotAccuracy(model=model, img_list=val_img_list, output_name='one_shot_accuracy'),
        ModelSaver(model_name="siamese_net", save_dir=model_dir, save_best='one_shot_accuracy', save_best_mode='max'),
        EarlyStopping(monitor="one_shot_accuracy", patience=20, compare='max', mode="eval")
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
