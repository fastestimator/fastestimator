#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import math
import tempfile

import numpy as np
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy, SuperLoss
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver, ImageSaver
from fastestimator.trace.metric import MCC
from fastestimator.trace.xai import LabelTracker


def big_lenet(classes=100, input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='swish', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='swish'))
    model.add(BatchNormalization())
    model.add(Dense(classes, activation='softmax'))
    return model


def corrupt_dataset(dataset, n_classes=100, corruption_fraction=0.4):
    # Keep track of which samples were corrupted for visualization later
    corrupted = [0 for _ in range(len(dataset))]
    # Perform the actual label corruption
    n_samples_per_class = len(dataset) // n_classes
    n_to_corrupt_per_class = math.floor(corruption_fraction * n_samples_per_class)
    n_corrupted = [0] * n_classes
    i = 0
    while any([elem < n_to_corrupt_per_class for elem in n_corrupted]):
        current_class = dataset[i]['y'].item()
        if n_corrupted[current_class] < n_to_corrupt_per_class:
            dataset[i]['y'] = (dataset[i]['y'] + np.random.randint(1, n_classes)) % n_classes
            n_corrupted[current_class] += 1
            corrupted[i] = 1
        i += 1
    # Put the corruption labels into the dataset for visualization
    dataset['data_labels'] = np.array(corrupted, dtype=np.int).reshape((len(dataset), 1))


def get_estimator(epochs=50,
                  batch_size=128,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = cifair100.load_data()

    # Add label noise to simulate real-world labeling problems
    corrupt_dataset(train_data)

    test_data = eval_data.split(range(len(eval_data) // 2))
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
        ])

    # step 2
    model = fe.build(model_fn=big_lenet, optimizer_fn='adam')
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SuperLoss(CrossEntropy(inputs=("y_pred", "y"), outputs="ce"), output_confidence="confidence"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3
    traces = [
        MCC(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        LabelTracker(metric="confidence",
                     label="data_labels",
                     label_mapping={
                         "Normal": 0, "Corrupted": 1
                     },
                     mode="train",
                     outputs="label_confidence"),
        ImageSaver(inputs="label_confidence", save_dir=save_dir, mode="train"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
