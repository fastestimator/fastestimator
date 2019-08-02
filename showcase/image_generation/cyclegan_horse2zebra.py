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
import os
import shutil
import time
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

import imageio
from cyclegan_model import Network
from fastestimator.estimator.estimator import Estimator
from fastestimator.estimator.trace import Trace, TrainLogger
from fastestimator.pipeline.dynamic.preprocess import AbstractPreprocessing
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.static.augmentation import AbstractAugmentation
from fastestimator.pipeline.static.filter import Filter


class GifGenerator(Trace):
    def __init__(self, save_path, export_name="anim.gif"):
        super().__init__()
        self.save_path = save_path
        self.prefix = "image_at_epoch_{0:04d}.png"
        self.export_name = os.path.join(self.save_path, export_name)

    def begin(self, mode):
        if mode == "eval":
            if not(os.path.exists(self.save_path)):
                os.makedirs(self.save_path)

    def on_batch_end(self, mode, logs):
        if mode == "eval":
            img = logs["prediction"]["Y_fake"]
            img = img[0, ...].numpy()
            img += 1
            img /= 2
            img *= 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(
                self.save_path,
                self.prefix.format(logs["epoch"])
            )
            cv2.imwrite(img_path, img.astype("uint8"))

    def end(self, mode):
        with imageio.get_writer(self.export_name, mode='I') as writer:
            filenames = glob(os.path.join(self.save_path, "*.png"))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 5*(i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

class Myrescale(AbstractPreprocessing):
    def transform(self, data, decoded_data=None):
        data = data.astype(np.float32)
        data = (data - 127.5) / 127.5
        return data

class MyImageReader(AbstractPreprocessing):
    def transform(self, path):
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return data

class RandomJitter(AbstractAugmentation):
    def __init__(self, mode="train"):
        self.mode = mode

    def transform(self, data):
        # resizing to 286 x 286 x 3
        data = tf.image.resize(
            data, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        data = tf.image.random_crop(data, size=[256, 256, 3])

        # random mirroring        
        data = tf.image.random_flip_left_right(data)

        return data

class my_filter_1(Filter):
    def __init__(self, mode="both"):
        self.mode = mode

    def filter_fn(self, dataset):
        return tf.equal(tf.reshape(dataset["label"], []), 1)


class my_filter_0(Filter):
    def __init__(self, mode="both"):
        self.mode = mode

    def filter_fn(self, dataset):
        return tf.equal(tf.reshape(dataset["label"], []), 0)


class my_pipeline(Pipeline):
    def final_transform(self, preprocessed_data):
        d1, d2 = preprocessed_data
        new_data = {}
        new_data["img_X"] = d1["img"]
        new_data["label_1"] = d1["label"]
        new_data["img_Y"] = d2["img"]
        new_data["label_2"] = d2["label"]
        return new_data

def get_estimator():
    # Step 1: Define Pipeline
    pipeline = my_pipeline(batch_size=1,
                           train_data="/root/data/public/horse2zebra/data.csv",
                           validation_data="/root/data/public/horse2zebra/val.csv",
                           feature_name=["img", "label"],
                           transform_dataset=[[MyImageReader(), Myrescale()], []],
                           transform_train=[[RandomJitter(mode="train")],
                                            []],
                           data_filter=[my_filter_0(), my_filter_1()])
    # Step2: Define Trace
    traces = [GifGenerator("/root/data/public/horse2zebra/images")]
    # Step3: Define Estimator
    estimator = Estimator(network=Network(),
                          pipeline=pipeline,
                          steps_per_epoch=1000,
                          validation_steps=1,
                          traces=traces,
                          epochs=10)
    return estimator
