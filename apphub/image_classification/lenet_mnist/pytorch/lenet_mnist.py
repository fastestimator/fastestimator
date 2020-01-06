"""This example showcase FastEstimator usage for pytorch users. In this file, we use data loader as data input.
"""
import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy

class MnistDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = {"x": self.x[idx], "y": self.y[idx]}
        return self.transform_data(data)

    def transform_data(self, data):
        data["x"] = np.expand_dims(data["x"], 0)
        data["x"] = np.float32(data["x"] / 255.0)
        data["y"] = np.int64(data["y"])
        return data

def get_dataloader(x, y, shuffle=True):
    dataset = MnistDataset(x=x, y=y)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=shuffle, num_workers=8)
    return loader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

    def num_flat_features(self, x):
        return np.prod(x.size()[1:])


def get_estimator():
    #step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    pipeline = fe.Pipeline(train_data=get_dataloader(x=x_train, y=y_train),
                           eval_data=get_dataloader(x=x_eval, y=y_eval, shuffle=False))
    #step 2
    model = fe.build(model=Net(), optimizer="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    #step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator

if __name__ == "__main__":
    est = get_estimator()
    est.fit()