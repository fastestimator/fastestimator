import tempfile
import unittest

import numpy as np

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch


class TestDataset(NumpyDataset):
    def __init__(self, data, var):
        super().__init__(data)
        self.var = var


class TestBatchDataset(unittest.TestCase):
    def test_save_and_load_state_with_batch_dataset_tf(self):
        def instantiate_system():
            system = sample_system_object()
            x_train = np.ones((2, 28, 28, 3))
            y_train = np.ones((2, ))
            ds = TestDataset(data={'x': x_train, 'y': y_train}, var=1)
            train_data = fe.dataset.BatchDataset(datasets=[ds, ds], num_samples=[1, 1])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=2)
            return system

        system = instantiate_system()

        # make some change
        new_var = 2
        system.pipeline.data["train"].datasets[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.data["train"].datasets[0].var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_batch_dataset_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            x_train = np.ones((2, 3, 28, 28))
            y_train = np.ones((2, ))
            ds = TestDataset(data={'x': x_train, 'y': y_train}, var=1)
            train_data = fe.dataset.BatchDataset(datasets=[ds, ds], num_samples=[1, 1])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=2)
            return system

        system = instantiate_system()

        # make some change
        new_var = 2
        system.pipeline.data["train"].datasets[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.data["train"].datasets[0].var

        self.assertEqual(loaded_var, new_var)
