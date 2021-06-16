# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import wget
from scipy import stats
from tensorflow.keras import Model, layers

import fastestimator as fe
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.search import GridSearch
from fastestimator.util import to_number
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress

# Predefined operation set
OPS = {
    'none':
    lambda inputs,
    n_filters,
    stride: _zero(inputs, n_filters),
    'avg_pool_3x3':
    lambda inputs,
    n_filters,
    stride: _pooling(inputs, n_filters, stride),
    'nor_conv_3x3':
    lambda inputs,
    n_filters,
    stride: _relu_conv_bn_block(inputs, n_filters, (3, 3), stride, "same", 1),
    'nor_conv_1x1':
    lambda inputs,
    n_filters,
    stride: _relu_conv_bn_block(inputs, n_filters, (1, 1), stride, "valid", 1),
    'skip_connect':
    lambda inputs,
    n_filters,
    stride: _identity(inputs)
    if stride == 1 and inputs.shape[-1] == n_filters else _factorize_reduce(inputs, n_filters, stride),
}


def _resnet_basic_block(inputs, n_filters, stride):
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    x = _relu_conv_bn_block(inputs, n_filters, kernel_size=3, stride=stride, padding="same", dilation=1)
    x = _relu_conv_bn_block(x, n_filters, kernel_size=3, stride=1, padding="same", dilation=1)

    if stride == 2:
        residual = layers.AveragePooling2D(pool_size=2, strides=stride, padding="valid")(inputs)
        residual = layers.Conv2D(n_filters, 1, 1, padding="valid", use_bias=False)(residual)
    elif inputs.shape[-1] != n_filters:
        residual = _relu_conv_bn_block(inputs, kernel_size=1, stride=1, padding="valid", dilation=1)
    else:
        residual = inputs

    return residual + x


def _relu_conv_bn_block(inputs, n_filters, kernel_size, stride, padding, dilation):
    x = layers.ReLU()(inputs)
    x = layers.Conv2D(n_filters, kernel_size, stride, padding=padding, dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return x


def _pooling(inputs, n_filters, stride):
    if inputs.shape[-1] != n_filters:
        inputs = _relu_conv_bn_block(inputs, n_filters, kernel_size=1, stride=1, padding="valid", dilation=1)

    x = layers.AveragePooling2D(pool_size=3, strides=stride, padding="same")(inputs)
    return x


def _identity(inputs):
    return inputs


def _zero(inputs, n_filters):
    inp_shape = inputs.shape

    if inp_shape[-1] == n_filters:
        return 0. * inputs
    else:
        inp_shape[-1] = n_filters
        return tf.zeros(inp_shape, inputs.dtype)


def _factorize_reduce(inputs, n_filters, stride):
    if stride == 2:
        filters_list = [n_filters // 2, n_filters - n_filters // 2]
        x = layers.ReLU()(inputs)
        y = tf.pad(inputs, [0, 0, 1, 1], mode="CONSTANT")
        x = layers.Conv2D(filters_list[0], kernel_size=1, stride=stride, padding="valid", use_bias=False)(x)
        y = layers.Conv2D(filters_list[1], kernel_size=1, stride=stride, padding="valid",
                          use_bias=False)(y[:, 1:, 1:, :])
        out = tf.cat([x, y], dim=1)
    elif stride == 1:
        out = layers.Conv2D(n_filters, kernel_size=1, stride=stride, padding="valid", use_bias=False)(inputs)
    else:
        raise ValueError('Invalid stride : {:}'.format(stride))

    out = layers.BatchNormalization(momentum=0.9)(out)
    return out


def str2structure(xstr):
    """Process the architecture string from NAS-Bench-201. Referenced from https://github.com/D-X-Y/AutoDL-Projects.
    """
    assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
    nodestrs = xstr.split('+')
    genotypes = []
    for node_str in nodestrs:
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs:
            assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = (xi.split('~') for xi in inputs)
        input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
        genotypes.append(input_infos)
    return genotypes


def _infer_cell(inputs, genotype, n_filters, stride):
    x_in = [inputs]

    for i in range(len(genotype)):
        node_info = genotype[i]
        if len(node_info) == 1:
            op_name, op_in = node_info[0]
            x = OPS[op_name](x_in[op_in], n_filters, stride) if op_in == 0 else OPS[op_name](x_in[op_in], n_filters, 1)
        else:
            x = layers.Add()([
                OPS[op_name](x_in[op_in], n_filters, stride) if op_in == 0 else OPS[op_name](x_in[op_in], n_filters, 1)
                for (op_name, op_in) in node_info
            ])
        x_in.append(x)

    return x


def nasbench_network(input_shape, genotype, C=16, N=5, num_classes=10):
    layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(C, kernel_size=3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)

    for (C_curr, reduction) in zip(layer_channels, layer_reductions):
        if reduction:
            x = _resnet_basic_block(x, n_filters=C_curr, stride=2)
        else:
            x = _infer_cell(x, genotype=genotype, n_filters=C_curr, stride=1)

    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, x)
    return model


def get_pipeline_data(batch_size=128):
    train_data, _ = cifar10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ])

    result = pipeline.get_results()
    return result


def score_fn(search_idx, uid, batch_data, config_info):
    config = config_info.loc[uid, :]
    nasbench201_model = nasbench_network((32, 32, 3),
                                         str2structure(config["architecture"]),
                                         config["C"],
                                         config["N"],
                                         10)
    feature_list = [layer.output for layer in nasbench201_model.layers if "re_lu" in layer.name]
    model = fe.build(model_fn=lambda: Model(nasbench201_model.input, feature_list), optimizer_fn="adam")

    # Only a single forward pass through the network is required
    relu_result = fe.backend.feed_forward(model, batch_data["x"])
    matrix = np.zeros((relu_result[0].shape[0], relu_result[0].shape[0]))
    for sample in relu_result:
        sample = to_number(sample)
        sample = sample.reshape((sample.shape[0], -1))
        x = (sample > 0.).astype(float)
        x_t = np.transpose(x)
        mat = x @ x_t
        mat2 = (1. - x) @ (1. - x_t)
        matrix = matrix + mat + mat2

    _, score = np.linalg.slogdet(matrix)
    return score


def fastestimator_run(batch_size=128, num_archs=1000, save_dir=tempfile.mkdtemp()):
    download_link = "https://github.com/fastestimator-util/fastestimator-misc/raw/master/resource/nasbench201_info.csv"
    uid_list = np.random.choice(15625, size=num_archs, replace=False)  # Select random set of networks

    wget.download(download_link, save_dir, bar=bar_custom)
    config_info = pd.read_csv(os.path.join(save_dir, "nasbench201_info.csv"))
    batch_data = get_pipeline_data(batch_size)

    search = GridSearch(
        score_fn=lambda search_idx,
        uid: score_fn(search_idx, uid, batch_data=batch_data, config_info=config_info),
        params={"uid": uid_list},
        best_mode="max")
    search.fit()

    best_results = search.get_best_results()
    score_list = [result[1] for result in search.get_search_results()]
    acc_list = [config_info.loc[i, :]["accuracy"] for i in uid_list]

    tau, _ = stats.kendalltau(acc_list, score_list)
    print("Kendall's Tau correlation coefficient: ", tau)

    print("Maximum accuracy among all the networks tested: ", np.max(acc_list))
    print("Params for best network: {}, best score: {} and corresponding accuracy: {}".format(
        best_results[0], best_results[1], config_info.loc[best_results[0]["uid"], :]["accuracy"]))
    print(
        "The best network is the top - {} network among the selected networks, based on trained performance (accuracy)".
        format(len(acc_list) - list(np.sort(acc_list)).index(config_info.loc[best_results[0]["uid"], :]["accuracy"])))


if __name__ == "__main__":
    fastestimator_run()
