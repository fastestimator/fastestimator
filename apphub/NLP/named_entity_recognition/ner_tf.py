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
from typing import Callable, Iterable, List, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel

import fastestimator as fe
from fastestimator.dataset.data import german_ner
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import PadSequence, Tokenize, WordtoId
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


def char2idx(data):
    tag2idx = {t: i for i, t in enumerate(data)}
    return tag2idx


class AttentionMask(NumpyOp):
    def forward(self, data, state):
        masks = [float(i > 0) for i in data]
        return np.array(masks)


class ReshapeOp(TensorOp):
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = "!infer"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        inp_shape = data.get_shape()
        if tf.keras.backend.ndim(data) < 3:
            return tf.reshape(data, [
                inp_shape[0] * inp_shape[1],
            ])
        else:
            return tf.reshape(data, [inp_shape[0] * inp_shape[1], inp_shape[2]])


def ner_model(max_len):
    token_inputs = Input((max_len), dtype=tf.int32, name='input_words')
    mask_inputs = Input((max_len), dtype=tf.int32, name='input_masks')
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    seq_output, _ = bert_model([token_inputs, mask_inputs])
    output = Dense(100, activation='relu')(seq_output)
    output = Dense(24, activation='softmax')(output)
    model = Model([token_inputs, mask_inputs], output)
    return model


def get_estimator(max_len=500,
                  epochs=10,
                  batch_size=64,
                  max_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):

    # step 1 prepare data
    train_data, eval_data, data_vocab, label_vocab = german_ner.load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tag2idx = char2idx(label_vocab)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Tokenize(inputs="x", outputs="x", tokenize_fn=tokenizer.tokenize),
            WordtoId(inputs="x", outputs="x", mapping=tokenizer.convert_tokens_to_ids),
            WordtoId(inputs="y", outputs="y", mapping=tag2idx),
            PadSequence(max_len=max_len, inputs="x", outputs="x"),
            PadSequence(max_len=max_len, value=tag2idx["O"], inputs="y", outputs="y"),
            AttentionMask(inputs="x", outputs="x_masks")
        ])

    # step 2. prepare model
    model = fe.build(model_fn=ner_model, optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs=["x", "x_masks"], outputs="y_pred"),
        ReshapeOp(inputs="y", outputs="y"),
        ReshapeOp(inputs="y_pred", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])

    traces = [Accuracy(true_key="y", pred_key="y_pred")]

    # step 3 prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
