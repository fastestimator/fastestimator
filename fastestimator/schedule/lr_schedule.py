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
import math
import os
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import wget

from fastestimator.backend._zscore import zscore
from fastestimator.util.traceability_util import traceable


def cosine_decay(time: int,
                 cycle_length: int,
                 init_lr: float,
                 min_lr: float = 1e-6,
                 start: int = 1,
                 cycle_multiplier: int = 1,
                 warmup: bool = False):
    """Learning rate cosine decay function (using half of cosine curve).

    This method is useful for scheduling learning rates which oscillate over time:
    ```python
    s = fe.schedule.LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])
    ```

    For more information, check out SGDR: https://arxiv.org/pdf/1608.03983.pdf.

    Args:
        time: The current step or epoch during training starting from 1.
        cycle_length: The decay cycle length.
        init_lr: Initial learning rate to decay from.
        min_lr: Minimum learning rate.
        start: The step or epoch to start the decay schedule.
        cycle_multiplier: The factor by which next cycle length will be multiplied.
        warmup: Whether to do a linear warmup from 0 up until `start'.

    Returns:
        lr: learning rate given current step or epoch.
    """
    if time < start:
        if warmup:
            lr = init_lr * time / start
        else:
            lr = init_lr
    else:
        time = time - start + 1
        if cycle_multiplier > 1:
            current_cycle_idx = math.ceil(
                math.log(time * (cycle_multiplier - 1) / cycle_length + 1) / math.log(cycle_multiplier)) - 1
            cumulative = cycle_length * (cycle_multiplier**current_cycle_idx - 1) / (cycle_multiplier - 1)
        elif cycle_multiplier == 1:
            current_cycle_idx = math.ceil(time / cycle_length) - 1
            cumulative = current_cycle_idx * cycle_length
        else:
            raise ValueError("multiplier must be at least 1")
        current_cycle_length = cycle_length * cycle_multiplier**current_cycle_idx
        time_in_cycle = (time - cumulative) / current_cycle_length
        lr = (init_lr - min_lr) / 2 * math.cos(time_in_cycle * math.pi) + (init_lr + min_lr) / 2
    return lr


@traceable()
class ARC:
    """A run-of-the-mill learning rate scheduler.

    Args:
        frequency: invoke frequency in terms of number of epochs.
    """
    def __init__(self, frequency: int = 3) -> None:
        self.frequency = frequency
        assert isinstance(self.frequency, int) and self.frequency > 0
        self.model = self._load_arc_model()
        self.lr_multiplier = {0: 1.618, 1: 1.0, 2: 0.618}
        self.train_loss_one_cycle = []
        self.eval_loss_one_cycle = []
        self.all_train_loss = deque([None] * 3, maxlen=3)
        self.all_eval_loss = deque([None] * 3, maxlen=3)
        self.all_train_lr = deque([None] * 3, maxlen=3)
        self.use_eval_loss = True

    def _load_arc_model(self) -> tf.keras.Model:
        arc_weight_path = self._initialize_arc_weight()
        with tf.device("cpu:0"):  # this ensures ARC model will not occupy gpu memory
            model = self.lstm_stacked()
            model.load_weights(arc_weight_path)
        return model

    @staticmethod
    def _initialize_arc_weight() -> str:
        arc_weight_path = os.path.join(str(Path.home()), "fastestimator_data", "arc_model", "arc.h5")
        if not os.path.exists(arc_weight_path):
            print("FastEstimator - Downloading ARC weights to {}".format(arc_weight_path))
            os.makedirs(os.path.split(arc_weight_path)[0], exist_ok=True)
            wget.download("https://github.com/fastestimator-util/fastestimator-misc/raw/master/resource/arc.h5",
                          out=arc_weight_path)
        return arc_weight_path

    @staticmethod
    def lstm_stacked() -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(300, 3)))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        model.add(tf.keras.layers.LSTM(64))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))
        return model

    def accumulate_single_train_loss(self, train_loss: float) -> None:
        self.train_loss_one_cycle.append(train_loss)

    def accumulate_single_eval_loss(self, eval_loss: float) -> None:
        self.eval_loss_one_cycle.append(eval_loss)

    def accumulate_all_lrs(self, lr: float) -> None:
        self.all_train_lr.append(lr)

    def gather_multiple_eval_losses(self) -> None:
        self.all_eval_loss.append(self.eval_loss_one_cycle)
        self.eval_loss_one_cycle = []

    def gather_multiple_train_losses(self) -> None:
        self.all_train_loss.append(self.train_loss_one_cycle)
        self.train_loss_one_cycle = []

    def predict_next_multiplier(self) -> float:
        train_loss, missing = self._merge_list(self.all_train_loss)
        train_loss = self._preprocess_train_loss(train_loss, missing)
        val_loss, missing = self._merge_list(self.all_eval_loss)
        val_loss = self._preprocess_val_loss(val_loss, missing)
        train_lr, missing = self._merge_list(self.all_train_lr)
        train_lr = self._preprocess_train_lr(train_lr, missing)
        model_inputs = np.concatenate((train_loss, val_loss, train_lr), axis=1)
        model_inputs = np.expand_dims(model_inputs, axis=0)
        with tf.device("cpu:0"):
            model_pred = self.model(model_inputs, training=False)
        action = np.argmax(model_pred)
        return self.lr_multiplier[action]

    def _preprocess_val_loss(self, val_loss: List[float], missing: int) -> np.ndarray:
        if val_loss:
            target_size = (3 - missing) * 100
            val_loss = np.array(val_loss, dtype="float32")
            val_loss = zscore(val_loss)
            val_loss = cv2.resize(val_loss, (1, target_size), interpolation=cv2.INTER_NEAREST)
            if val_loss.size < 300:
                val_loss = np.pad(val_loss, ((300 - val_loss.size, 0), (0, 0)), mode='constant', constant_values=0.0)
        else:
            val_loss = np.zeros([300, 1], dtype="float32")
        return val_loss

    def _preprocess_train_lr(self, train_lr: List[float], missing: int) -> np.ndarray:
        target_size = (3 - missing) * 100
        train_lr = np.array(train_lr) / train_lr[0]
        train_lr = cv2.resize(train_lr, (1, target_size), interpolation=cv2.INTER_NEAREST)
        if train_lr.size < 300:
            train_lr = np.pad(train_lr, ((300 - train_lr.size, 0), (0, 0)), mode='constant', constant_values=1.0)
        return train_lr

    def _preprocess_train_loss(self, train_loss: List[float], missing: int) -> np.ndarray:
        target_size = (3 - missing) * 100
        train_loss = np.array(train_loss, dtype="float32")
        train_loss = cv2.resize(train_loss, (1, target_size))
        train_loss = zscore(train_loss)
        if train_loss.size < 300:
            train_loss = np.pad(train_loss, ((300 - train_loss.size, 0), (0, 0)), mode='constant', constant_values=0.0)
        return train_loss

    def _merge_list(self, data: List[Union[None, float, List[float]]]) -> Tuple[List[float], int]:
        output = []
        missing = 0
        for item in data:
            if isinstance(item, list):
                output.extend(item)
            elif item:
                output.append(item)
            else:
                missing += 1
        return output, missing
