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
import inspect
import os
from collections import deque
from typing import Callable, List, Optional, Tuple, Union

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.feed_forward import feed_forward
from fastestimator.backend.get_lr import get_lr
from fastestimator.backend.set_lr import set_lr
from fastestimator.backend.zscore import zscore
from fastestimator.network import build
from fastestimator.summary.system import System
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class LRScheduler(Trace):
    """Learning rate scheduler trace that changes the learning rate while training.

    This class requires an input function which takes either 'epoch' or 'step' as input:
    ```python
    s = LRScheduler(model=model, lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])  # Learning rate will change based on step
    s = LRScheduler(model=model, lr_fn=lambda epoch: fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])  # Learning rate will change based on epoch
    s = LRScheduler(model=model, lr_fn='arc')
    fe.Estimator(..., traces=[s])  # Learning rate will change based on epoch using ARC algorithm
    ```

    Args:
        model: A model instance compiled with fe.build.
        lr_fn: A lr scheduling function that takes either 'epoch' or 'step' as input, or the string 'arc'.
        frequency: The frequency with which to invoke the `lr_fn`, or None for default.

    Raises:
        AssertionError: If the `lr_fn` is not configured properly.
    """
    system: System

    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 lr_fn: Union[Callable[[int], float], str],
                 frequency: Optional[int] = None) -> None:
        self.model = model
        self.lr_fn = lr_fn
        self.frequency = frequency
        assert hasattr(lr_fn, "__call__") or lr_fn == "arc", "lr_fn must be a function or 'arc'"
        if frequency:
            assert isinstance(frequency, int) and frequency > 0, "frequency must be a positive integer"
        if lr_fn == "arc":
            self.lr_fn = ARC(os.path.join(os.path.split(fe.__file__)[0], "resources", "arc.h5"))
            self.schedule_mode = "epoch"
        else:
            arg = list(inspect.signature(lr_fn).parameters.keys())
            assert len(arg) == 1 and arg[0] in {"step", "epoch"}, "the lr_fn input arg must be either 'step' or 'epoch'"
            self.schedule_mode = arg[0]
            if self.frequency is None:
                self.frequency = 1
        super().__init__(mode=None, outputs=self.model.model_name + "_lr")

    def on_begin(self, data: Data) -> None:
        if isinstance(self.lr_fn, ARC):
            assert len(self.model.loss_name) == 1, "arc can only work with single model loss"
            self.lr_fn.use_eval_loss = "eval" in self.system.pipeline.data
            if self.frequency is None:
                self.frequency = np.clip(int(np.floor(self.system.total_epochs / 10)), 1, 10)

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.mode == "train" and self.schedule_mode == "epoch" and (
                self.system.epoch_idx % self.frequency == 1 or self.frequency == 1):
            if isinstance(self.lr_fn, ARC):
                if self.system.epoch_idx > 1:
                    multiplier = self.lr_fn.predict_next_multiplier()
                    new_lr = np.float32(get_lr(model=self.model) * multiplier)
                    set_lr(self.model, new_lr)
                    print("FastEstimator-ARC: Multiplying LR by {}".format(multiplier))
            else:
                new_lr = np.float32(self.lr_fn(self.system.epoch_idx))
                set_lr(self.model, new_lr)

    def on_batch_begin(self, data: Data) -> None:
        if self.system.mode == "train" and self.schedule_mode == "step" and \
            (self.system.global_step % self.frequency == 1 or self.frequency == 1):
            new_lr = np.float32(self.lr_fn(self.system.global_step))
            set_lr(self.model, new_lr)

    def on_batch_end(self, data: Data) -> None:
        if self.system.mode == "train" and isinstance(self.lr_fn, ARC):
            self.lr_fn.accumulate_single_train_loss(data[min(self.model.loss_name)].numpy())
        if self.system.mode == "train" and self.system.log_steps and (
                self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1):
            current_lr = np.float32(get_lr(self.model))
            data.write_with_log(self.outputs[0], current_lr)

    def on_epoch_end(self, data: Data) -> None:
        if self.system.mode == "eval" and isinstance(self.lr_fn, ARC):
            self.lr_fn.accumulate_single_eval_loss(data[min(self.model.loss_name)])
            if self.system.epoch_idx % self.frequency == 0:
                self.lr_fn.gather_multiple_eval_losses()
        if self.system.mode == "train" and isinstance(self.lr_fn, ARC) and self.system.epoch_idx % self.frequency == 0:
            self.lr_fn.accumulate_all_lrs(get_lr(model=self.model))
            self.lr_fn.gather_multiple_train_losses()


@traceable()
class ARC:
    def __init__(self, weights_path: str) -> None:
        with tf.device("cpu:0"):
            self.model = build(model_fn=self.lstm_stacked, optimizer_fn=None, weights_path=weights_path)
        self.lr_multiplier = {0: 1.618, 1: 1.0, 2: 0.618}
        self.train_loss_one_cycle = []
        self.eval_loss_one_cycle = []
        self.all_train_loss = deque([None] * 3, maxlen=3)
        self.all_eval_loss = deque([None] * 3, maxlen=3)
        self.all_train_lr = deque([None] * 3, maxlen=3)
        self.use_eval_loss = True

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

    def lstm_stacked(self) -> tf.keras.Model:
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

    def predict_next_multiplier(self) -> float:
        train_loss, missing = self._merge_list(self.all_train_loss)
        train_loss = self._preprocess_train_loss(train_loss, missing)
        val_loss, _ = self._merge_list(self.all_eval_loss)
        val_loss = self._preprocess_val_loss(val_loss)
        train_lr, _ = self._merge_list(self.all_train_lr)
        train_lr = self._preprocess_train_lr(train_lr)
        model_inputs = np.concatenate((train_loss, val_loss, train_lr), axis=1)
        model_inputs = np.expand_dims(model_inputs, axis=0)
        with tf.device("cpu:0"):
            model_pred = feed_forward(model=self.model, x=model_inputs, training=False)
        action = np.argmax(model_pred)
        return self.lr_multiplier[action]

    def _preprocess_val_loss(self, val_loss: List[float]) -> np.ndarray:
        if val_loss:
            val_loss = zscore(np.array(val_loss))
            val_loss = cv2.resize(val_loss, (1, 300), interpolation=cv2.INTER_NEAREST)
        else:
            val_loss = np.zeros([300, 1], dtype="float32")
        return val_loss

    def _preprocess_train_lr(self, train_lr: List[float]) -> np.ndarray:
        train_lr = np.array(train_lr) / train_lr[-1]
        train_lr = cv2.resize(train_lr, (1, 300), interpolation=cv2.INTER_NEAREST)
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
