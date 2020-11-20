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
import random
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps, ImageTransform
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import OneOf
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


def residual(x, num_channel):
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def my_model():
    # prep layers
    inp = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer1
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    # layer2
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer3
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    # layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    return model


class Rotate(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.degree = level * 3.0

    def forward(self, data, state):
        im = Image.fromarray(data)
        degree = self.degree * random.choice([1.0, -1.0])
        im = im.rotate(degree)
        return np.asarray(im)


class Identity(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)


class AutoContrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        im = Image.fromarray(data)
        im = ImageOps.autocontrast(im)
        return np.copy(np.asarray(im))


class Equalize(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        im = Image.fromarray(data)
        im = ImageOps.equalize(im)
        return np.copy(np.asarray(im))


class Posterize(NumpyOp):
    # resuce the number of bits for each channel, this may be inconsistent with original implementation
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.bits = 8 - int((level / 10) * 4)

    def forward(self, data, state):
        im = Image.fromarray(data)
        im = ImageOps.posterize(im, self.bits)
        return np.copy(np.asarray(im))


class Solarize(NumpyOp):
    # this may be inconsistent with original implementation
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.threshold = 256 - int(level * 25.6)

    def forward(self, data, state):
        data = np.where(data < self.threshold, data, 255 - data)
        return data


class Sharpness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff = 0.09 * level

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + self.diff * random.choice([1.0, -1.0])
        im = ImageEnhance.Sharpness(im).enhance(factor)
        return np.copy(np.asarray(im))


class Contrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff = 0.09 * level

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + self.diff * random.choice([1.0, -1.0])
        im = ImageEnhance.Contrast(im).enhance(factor)
        return np.copy(np.asarray(im))


class Color(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff = 0.09 * level

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + self.diff * random.choice([1.0, -1.0])
        im = ImageEnhance.Color(im).enhance(factor)
        return np.copy(np.asarray(im))


class Brightness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff = 0.09 * level

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + self.diff * random.choice([1.0, -1.0])
        im = ImageEnhance.Brightness(im).enhance(factor)
        return np.copy(np.asarray(im))


class ShearX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level * 0.03

    def forward(self, data, state):
        im = Image.fromarray(data)
        shear_coeff = self.shear_coef * random.choice([1.0, -1.0])
        width, height = im.size
        xshift = int(round(self.shear_coef * width))
        new_width = width + xshift
        im = im.transform((new_width, height),
                          ImageTransform.AffineTransform(
                              (1.0, shear_coeff, -xshift if shear_coeff > 0 else 0.0, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        if self.shear_coef > 0:
            im = im.resize((width, height))
        return np.copy(np.asarray(im))


class ShearY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level * 0.03

    def forward(self, data, state):
        im = Image.fromarray(data)
        shear_coeff = self.shear_coef * random.choice([1.0, -1.0])
        width, height = im.size
        yshift = int(round(self.shear_coef * height))
        newheight = height + yshift
        im = im.transform((width, newheight),
                          ImageTransform.AffineTransform(
                              (1.0, 0.0, 0.0, shear_coeff, 1.0, -yshift if shear_coeff > 0 else 0.0)),
                          resample=Image.BICUBIC)
        if self.shear_coef > 0:
            im = im.resize((width, height))
        return np.copy(np.asarray(im))


class TranslateX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = int(self.level / 10 * width / 3 * random.choice([1.0, -1.0]))
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, displacement, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


class TranslateY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = int(self.level / 10 * height / 3 * random.choice([1.0, -1.0]))
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, 0.0, 0.0, 1.0, displacement)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


def get_estimator(level,
                  num_augment,
                  epochs=24,
                  batch_size=512,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None):
    assert 0 <= level <= 10, "the level should be between 0 and 10"
    train_data, test_data = load_data()
    aug_ops = [
        OneOf(
            Rotate(level=level, inputs="x", outputs="x", mode="train"),
            Identity(level=level, inputs="x", outputs="x", mode="train"),
            AutoContrast(level=level, inputs="x", outputs="x", mode="train"),
            Equalize(level=level, inputs="x", outputs="x", mode="train"),
            Posterize(level=level, inputs="x", outputs="x", mode="train"),
            Solarize(level=level, inputs="x", outputs="x", mode="train"),
            Sharpness(level=level, inputs="x", outputs="x", mode="train"),
            Contrast(level=level, inputs="x", outputs="x", mode="train"),
            Color(level=level, inputs="x", outputs="x", mode="train"),
            Brightness(level=level, inputs="x", outputs="x", mode="train"),
            ShearX(level=level, inputs="x", outputs="x", mode="train"),
            ShearY(level=level, inputs="x", outputs="x", mode="train"),
            TranslateX(level=level, inputs="x", outputs="x", mode="train"),
            TranslateY(level=level, inputs="x", outputs="x", mode="train"),
        ) for _ in range(num_augment)
    ]
    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=aug_ops + [
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ])
    model = fe.build(model_fn=my_model, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=Accuracy(true_key="y", pred_key="y_pred"),
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    result = np.zeros(shape=(10, 10))
    for level in range(1, 11):
        for num_augment in range(1, 11):
            print("Trying level {} and num_augment {}".format(level, num_augment))
            est = get_estimator(level=level, num_augment=num_augment, epochs=50)
            est.fit(summary="exp")
            hist = est.test(summary="exp")
            result[level - 1, num_augment - 1] = hist.history["test"]["accuracy"][4900]
    np.save("result.npy", result)
