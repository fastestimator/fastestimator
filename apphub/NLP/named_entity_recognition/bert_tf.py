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
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel

import fastestimator as fe
from fastestimator.dataset.data import mitmovie_ner
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import PadSequence, Tokenize, WordtoId
from fastestimator.op.tensorop import Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def char2idx(data):
    tag2idx = {t: i for i, t in enumerate(data)}
    return tag2idx


class AttentionMask(NumpyOp):
    def forward(self, data, state):
        masks = [float(i > 0) for i in data]
        return np.array(masks)


def ner_model(max_len, pretrained_model, label_vocab):
    token_inputs = Input((max_len), dtype=tf.int32, name='input_words')
    mask_inputs = Input((max_len), dtype=tf.int32, name='input_masks')
    bert_model = TFBertModel.from_pretrained(pretrained_model)
    bert_output = bert_model(token_inputs, attention_mask=mask_inputs) # use the last hidden state
    output = Dense(len(label_vocab) + 1, activation='softmax')(bert_output[0])
    model = Model([token_inputs, mask_inputs], output)
    return model


def get_estimator(max_len=20,
                  epochs=10,
                  batch_size=64,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  pretrained_model='bert-base-uncased',
                  data_dir=None):
    # step 1 prepare data
    train_data, eval_data, data_vocab, label_vocab = mitmovie_ner.load_data(root_dir=data_dir)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
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
            PadSequence(max_len=max_len, value=len(tag2idx), inputs="y", outputs="y"),
            AttentionMask(inputs="x", outputs="x_masks")
        ])

    # step 2. prepare model
    model = fe.build(model_fn=lambda: ner_model(max_len, pretrained_model, label_vocab),
                     optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs=["x", "x_masks"], outputs="y_pred"),
        Reshape(inputs="y", outputs="y", shape=(-1, )),
        Reshape(inputs="y_pred", outputs="y_pred", shape=(-1, len(label_vocab) + 1)),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])

    traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=save_dir)]

    # step 3 prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
