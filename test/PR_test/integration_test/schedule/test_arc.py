import math
import unittest

import numpy as np

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet as LeNet_torch
from fastestimator.architecture.tensorflow import LeNet as LeNet_tf
from fastestimator.backend import get_lr
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import ARC
from fastestimator.trace.adapt import LRScheduler


class TestLRScheduler(unittest.TestCase):

    @staticmethod
    def create_estimator_for_arc(model, use_eval, axis):
        train_data, eval_data = mnist.load_data()
        pipeline = fe.Pipeline(train_data=train_data,
                               eval_data=eval_data if use_eval else None,
                               batch_size=8,
                               ops=[ExpandDims(inputs="x", outputs="x", axis=axis), Minmax(inputs="x", outputs="x")])
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])
        estimator = fe.Estimator(pipeline=pipeline,
                                 network=network,
                                 epochs=2,
                                 traces=LRScheduler(model=model, lr_fn=ARC(1)),
                                 max_train_steps_per_epoch=10)
        return estimator

    def test_tf_model_arc_train_eval(self):
        model_tf = fe.build(model_fn=LeNet_tf, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=True, axis=-1)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_tf_model_arc_train_only(self):
        model_tf = fe.build(model_fn=LeNet_tf, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=False, axis=-1)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = np.round(lr_after / lr_before, 3)
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_torch_model_arc_train_eval(self):
        model_tf = fe.build(model_fn=LeNet_torch, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=True, axis=0)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_torch_model_arc_train_only(self):
        model_tf = fe.build(model_fn=LeNet_torch, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=False, axis=0)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)
