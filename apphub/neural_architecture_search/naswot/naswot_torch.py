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
"""
The FastEstimator implementation of Neural Architecture Search without Training on CIFAR10.
The model architecture implementation took reference from https://github.com/D-X-Y/AutoDL-Projects.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wget
from scipy import stats

import fastestimator as fe
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize
from fastestimator.search import GridSearch
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress

# Predefined operation set
OPS = {
    'none':
    lambda C_in,
    C_out,
    stride,
    affine,
    track_running_stats: Zero(C_in, C_out, stride),
    'avg_pool_3x3':
    lambda C_in,
    C_out,
    stride,
    affine,
    track_running_stats: POOLING(C_in, C_out, stride, affine, track_running_stats),
    'nor_conv_3x3':
    lambda C_in,
    C_out,
    stride,
    affine,
    track_running_stats: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1), affine, track_running_stats),
    'nor_conv_1x1':
    lambda C_in,
    C_out,
    stride,
    affine,
    track_running_stats: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1), affine, track_running_stats),
    'skip_connect':
    lambda C_in,
    C_out,
    stride,
    affine,
    track_running_stats: Identity()
    if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))

    def forward(self, x):
        return self.op(x)


class POOLING(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, inputs):
        if self.preprocess: x = self.preprocess(inputs)
        else: x = inputs
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1: return x.mul(0.)
            else: return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
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


class InferCell(nn.Module):
    def __init__(self, genotype, C_in, C_out, stride):
        super(InferCell, self).__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        for i in range(len(genotype)):
            node_info = genotype[i]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](C_in, C_out, stride, True, True)
                else:
                    layer = OPS[op_name](C_out, C_out, 1, True, True)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def forward(self, inputs):
        nodes = [inputs]
        for (node_layers, node_innods) in zip(self.node_IX, self.node_IN):
            node_feature = sum(self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods))
            nodes.append(node_feature)
        return nodes[-1]


class NasbenchNetwork(nn.Module):
    def __init__(self, genotype, C, N, num_classes, batch_size=128):
        super(NasbenchNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for (C_curr, reduction) in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.relu_out = {}

        for _, module in self.named_modules():
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(self.relu_hook)

    def relu_hook(self, module, inp, out):
        try:
            self.relu_out[inp[0].device].append(out.view(out.size(0), -1))
        except:
            self.relu_out[inp[0].device] = [out.view(out.size(0), -1)]

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


def get_pipeline_data(batch_size=128):
    train_data, _ = cifar10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            ChannelTranspose(inputs="x", outputs="x")
        ])

    result = pipeline.get_results()
    return result


def score_fn(search_idx, uid, batch_data, config_info, batch_size):
    config = config_info.loc[uid, :]
    model = fe.build(
        model_fn=lambda: NasbenchNetwork(
            str2structure(config["architecture"]), config["C"], config["N"], 10, batch_size),
        optimizer_fn=None)

    if torch.cuda.is_available():
        batch_data["x"] = batch_data["x"].to("cuda")
        model = model.to("cuda")

    _ = fe.backend.feed_forward(model, batch_data["x"], training=False)

    if torch.cuda.device_count() > 1:
        model = model.module

    key_set = []
    for key in model.relu_out.keys():
        key_set.append(key)

    matrix = np.zeros((batch_size, batch_size))
    for i in range(len(model.relu_out[key_set[0]])):
        x = np.concatenate([(model.relu_out[key][i] > 0).float().cpu().numpy() for key in key_set], axis=0)
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
        eval_fn=lambda search_idx,
        uid: score_fn(search_idx, uid, batch_data=batch_data, config_info=config_info),
        params={"uid": uid_list},
        best_mode="max")
    search.fit()

    best_results = search.get_best_results()
    score_list = [result['result']['value'] for result in search.get_search_summary()]
    acc_list = [config_info.loc[i, :]["accuracy"] for i in uid_list]

    tau, _ = stats.kendalltau(acc_list, score_list)
    print("Kendall's Tau correlation coefficient: ", tau)

    print("Maximum accuracy among all the networks tested: ", np.max(acc_list))
    print("Params for best network: {}, best score: {} and corresponding accuracy: {}".format(
        best_results['param'],
        best_results['result']['value'],
        config_info.loc[best_results['param']["uid"], :]["accuracy"]))
    print(
        "The best network is the top - {} network among the selected networks, based on trained performance (accuracy)".
        format(
            len(acc_list) -
            list(np.sort(acc_list)).index(config_info.loc[best_results['param']["uid"], :]["accuracy"])))


if __name__ == "__main__":
    fastestimator_run()
