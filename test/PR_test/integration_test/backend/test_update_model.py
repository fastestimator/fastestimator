import unittest
from copy import deepcopy

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import OneLayerTorchModel, one_layer_tf_model


class TestUpdateModel(unittest.TestCase):
    def test_tf_model_with_get_gradient(self):
        def update(x, model):
            with tf.GradientTape(persistent=True) as tape:
                y = fe.backend.feed_forward(model, x)
                gradient = fe.backend.get_gradient(target=y, sources=model.trainable_variables, tape=tape)
                fe.backend.update_model(model, gradients=gradient)
                return gradient

        lr = 0.1
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn=lambda: tf.optimizers.SGD(lr))
        init_weights = [x.numpy() for x in model.trainable_variables]
        x = tf.constant([[1, 1, 1], [1, 1, 1]])
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            gradients = strategy.run(update, args=(x, model))
            gradients = [x.values[0].numpy() for x in gradients]  # from PerReplica to numpy
            # only take the first because other are the same
        else:
            gradients = update(x, model)
            gradients = [x.numpy() for x in gradients]
        gradients_factor = max(torch.cuda.device_count(), 1)  # if multi-gpu, the gradient will accumulate
        new_weights = [x.numpy() for x in model.trainable_variables]
        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr * gradients_factor
            self.assertTrue(np.allclose(new_w_ans, new_w))

    def test_torch_model_with_get_gradient(self):
        lr = 0.1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = fe.build(model_fn=OneLayerTorchModel,
                         optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=lr)).to(device)
        init_weights = [deepcopy(x).cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        x = torch.tensor([1.0, 1.0, 1.0]).to(torch.device(device))
        y = fe.backend.feed_forward(model.module if torch.cuda.device_count() > 1 else model, x)

        gradients = fe.backend.get_gradient(target=y, sources=[x for x in model.parameters() if x.requires_grad])
        fe.backend.update_model(model, gradients=gradients)

        gradients = [x.cpu().detach().numpy() for x in gradients]
        new_weights = [x.cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr
            self.assertTrue(np.allclose(new_w_ans, new_w))

    def test_tf_model_with_arbitrary_gradient(self):
        def update(gradients, model):
            fe.backend.update_model(model, gradients=gradients)

        lr = 0.1
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn=lambda: tf.optimizers.SGD(lr))
        init_weights = [x.numpy() for x in model.trainable_variables]
        gradients = [tf.constant([[1.0], [1.0], [1.0]])]

        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            strategy.run(update, args=(gradients, model))
        else:
            update(gradients, model)
        gradients = [x.numpy() for x in gradients]  # tensor to numpy
        new_weights = [x.numpy() for x in model.trainable_variables]
        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr
            self.assertTrue(np.allclose(new_w_ans, new_w))

    def test_torch_model_with_arbitrary_gradient(self):
        lr = 0.1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = fe.build(model_fn=OneLayerTorchModel,
                         optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=lr)).to(device)
        init_weights = [deepcopy(x).cpu().detach().numpy() for x in model.parameters() if x.requires_grad]
        gradients = [torch.tensor([[1.0, 1.0, 1.0]]).to(torch.device(device))]
        fe.backend.update_model(model, gradients=gradients)

        gradients = [x.cpu().detach().numpy() for x in gradients]
        new_weights = [x.cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr
            self.assertTrue(np.allclose(new_w_ans, new_w))
