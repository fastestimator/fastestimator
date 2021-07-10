# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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
import time
import unittest

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe
from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.schedule import EpochScheduler
from fastestimator.test.unittest_util import is_equal


class SampleNumpyOp(NumpyOp):
    def forward(self, data, state):
        return data


class SampleTensorOp(TensorOp):
    def forward(self, data, state):
        return data


class NumpyOpAdd1(NumpyOp):
    def forward(self, data, state):

        return data + 1


class ListData(Dataset):
    def __init__(self, ds, key1="x", key2="y"):
        self.ds = ds
        self.key1 = key1
        self.key2 = key2

    def __getitem__(self, idx):
        return [{self.key1: self.ds[idx]["x"], self.key2: self.ds[idx]["y"]} for _ in range(5)]

    def __len__(self):
        return len(self.ds)


class TorchCustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}


def get_sample_tf_dataset():
    x_train = np.array([[x] for x in range(100)], dtype=np.float32)
    y_train = np.array([[x] for x in range(-99, 1)], dtype=np.float32)
    train_data = {"x": x_train, "y": y_train}
    dataset_tf = tf.data.Dataset.from_tensor_slices(train_data)
    dataset_tf = dataset_tf.batch(4)
    return dataset_tf


def get_sample_torch_dataset():
    x_train = np.array([[x] for x in range(100)], dtype=np.float32)
    y_train = np.array([[x] for x in range(-99, 1)], dtype=np.float32)
    train_data = {"x": x_train, "y": y_train}
    return TorchCustomDataset(train_data)


def get_sample_torch_dataloader():
    x_train = np.array([[x] for x in range(100)], dtype=np.float32)
    y_train = np.array([[x] for x in range(-99, 1)], dtype=np.float32)
    train_data = {"x": x_train, "y": y_train}
    dataset_torch = TorchCustomDataset(train_data)
    dataloader_torch = DataLoader(dataset_torch, batch_size=4)
    return dataloader_torch


class TestPipelineInit(unittest.TestCase):
    """ This test cover:
    * fe.pipeline._verify_inputs
    * fe.pipeline._verify_dataset

    This test has dependency:
    * fe.schedule.schedule.get_current_items
    * fe.schedule.schedule.EpochSceduler
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_tf_dataset = get_sample_tf_dataset()
        cls.sample_torch_dataset = get_sample_torch_dataset()
        cls.sample_torch_dataloader = get_sample_torch_dataloader()
        cls.sample_numpy_op = SampleNumpyOp(inputs="x", outputs="x")
        cls.sample_tensor_op = SampleTensorOp(inputs="x", outputs="x")

    def test_pipeline_init_tf_dataset_torch_dataloader_have_op_batch_size_num_process(self):
        dataset = {"tf_dataset": self.sample_tf_dataset, "dataloader": self.sample_torch_dataloader}

        for data_type, data in dataset.items():
            with self.subTest("{} with numpyop".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, ops=[self.sample_numpy_op])

            with self.subTest("{} with batch_size not None".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, batch_size=10)

            with self.subTest("{} with num_process not None".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, num_process=1)

    def test_pipeline_init_tf_dataset_torch_dataloader_scheduler_have_op_batch_size_num_process(self):
        dataset = {"tf_dataset": self.sample_tf_dataset, "dataloader": self.sample_torch_dataloader}

        for data_type, data in dataset.items():
            scheduler_dataset = EpochScheduler(epoch_dict={1: data, 2: None})
            with self.subTest("{} with numpyop".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=scheduler_dataset,
                                           eval_data=scheduler_dataset,
                                           test_data=scheduler_dataset,
                                           ops=[self.sample_numpy_op])

            with self.subTest("{} with batch_size not None".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=scheduler_dataset,
                                           eval_data=scheduler_dataset,
                                           test_data=scheduler_dataset,
                                           batch_size=10)

            with self.subTest("{} with num_process not None".format(data_type)):
                with self.assertRaises(AssertionError):
                    pipeline = fe.Pipeline(train_data=scheduler_dataset,
                                           eval_data=scheduler_dataset,
                                           test_data=scheduler_dataset,
                                           num_process=1)

    def test_pipeline_init_torch_dataset_have_op_batch_size_num_process(self):
        data = self.sample_torch_dataset

        with self.subTest("with numpyop"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, ops=[self.sample_numpy_op])
            except:
                self.fail("exception occur")

        with self.subTest("with batch_size not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, batch_size=10)
            except:
                self.fail("exception occur")

        with self.subTest("with num_process not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, num_process=1)
            except:
                self.fail("exception occur")

    def test_pipeline_init_torch_dataset_scheduler_have_op_batch_size_num_process(self):
        data = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 2: None})

        with self.subTest("with numpyop"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, ops=[self.sample_numpy_op])
            except:
                self.fail("exception occur")

        with self.subTest("with batch_size not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, batch_size=10)
            except:
                self.fail("exception occur")

        with self.subTest("with num_process not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, num_process=1)
            except:
                self.fail("exception occur")

    def test_pipeline_init_all_dataset_no_op_batch_size_num_process(self):
        dataset = {
            "tf_dataset": self.sample_tf_dataset,
            "dataloader": self.sample_torch_dataloader,
            "torch_dataset": self.sample_torch_dataset
        }

        for data_type, data in dataset.items():
            with self.subTest("{}".format(data_type)):
                try:
                    pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data)
                except:
                    self.fail("exception occur")

    def test_pipeline_init_torch_dataset_with_tensorop(self):
        data = self.sample_torch_dataset
        with self.subTest("all tensorop"):
            with self.assertRaises(AssertionError):
                pipeline = fe.Pipeline(train_data=data,
                                       eval_data=data,
                                       test_data=data,
                                       ops=[self.sample_tensor_op, self.sample_tensor_op])
        with self.subTest("mixed tensorop numpyop"):
            with self.assertRaises(AssertionError):
                pipeline = fe.Pipeline(train_data=data,
                                       eval_data=data,
                                       test_data=data,
                                       ops=[self.sample_tensor_op, self.sample_numpy_op])


class TestPipelineGetModes(unittest.TestCase):
    """This test include:
    * fe.pipeline.Pipeline.get_modes
    * fe.schedule.schedule.EpochSceduler
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_torch_dataset = get_sample_torch_dataset()

    def test_pipeline_get_modes_no_scheduler(self):
        with self.subTest("train_data"):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)
            modes = pipeline.get_modes()
            self.assertEqual(modes, {"train"})

        with self.subTest("eval_data, test_data"):
            pipeline = fe.Pipeline(eval_data=self.sample_torch_dataset, test_data=self.sample_torch_dataset)
            modes = pipeline.get_modes()
            self.assertEqual(modes, {"eval", "test"})

    def test_pipeline_get_mode_epoch_scheduler(self):
        train_data = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 2: None})
        eval_data = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 3: None})
        test_data = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 4: None})
        pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data)

        with self.subTest(epoch=1):
            modes = pipeline.get_modes(epoch=1)
            self.assertEqual(modes, {"train", "eval", "test"})

        with self.subTest(epoch=2):
            modes = pipeline.get_modes(epoch=2)
            self.assertEqual(modes, {"eval", "test"})

        with self.subTest(epoch=3):
            modes = pipeline.get_modes(epoch=3)
            self.assertEqual(modes, {"test"})

        with self.subTest(epoch=4):
            modes = pipeline.get_modes(epoch=4)
            self.assertEqual(modes, set())


class TestPipelineGetEpochsWithData(unittest.TestCase):
    """This test include:
    * fe.pipeline.Pipeline.get_epochs_with_data
    * fe.schedule.schedule.EpochSceduler
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_torch_dataset = get_sample_torch_dataset()

    def test_pipeline_get_epochs_with_data_no_scheduler(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)

        with self.subTest("mode has dataset"):
            epochs = pipeline.get_epochs_with_data(total_epochs=5, mode="train")
            self.assertEqual(epochs, {1, 2, 3, 4, 5})

        with self.subTest("mode has no dataset"):
            with self.assertRaises(KeyError):
                epochs = pipeline.get_epochs_with_data(total_epochs=5, mode="eval")

    def test_pipeline_get_epochs_with_data_with_scheduler(self):
        dataset = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 3: None})
        pipeline = fe.Pipeline(train_data=dataset)
        epochs = pipeline.get_epochs_with_data(total_epochs=5, mode="train")
        self.assertEqual(epochs, {1, 2})


class TestPipelineBechmark(unittest.TestCase):
    """ This test has dependency on:
    * fe.pipeline.Pipeline.get_loader
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_tf_dataset = get_sample_tf_dataset()
        cls.sample_torch_dataset = get_sample_torch_dataset()
        cls.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_bechmark_smoke(self):
        dataset = {
            "tf_dataset": self.sample_tf_dataset,
            "torch_dataset": self.sample_torch_dataset,
            "torch_dataloader": self.sample_torch_dataloader
        }

        for data_type, data in dataset.items():
            with self.subTest("{}".format(data_type)):
                pipeline = fe.Pipeline(train_data=data)

                try:
                    pipeline.benchmark()
                except:
                    self.fail("exception occur")
                time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.


class TestPipelineTransform(unittest.TestCase):
    """ This test has dependency on:
    * fe.schedule.schedule.get_current_items
    * fe.op.numpy.numpy.forward_numpy
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_data = {"x": np.array([1, 2, 3], dtype=np.float32)}
        cls.sample_dataset = get_sample_torch_dataset()

    def test_pipeline_transform_no_ops(self):
        pipeline = fe.Pipeline()
        data = pipeline.transform(data=self.sample_data, mode="train")
        ans = {"x": np.array([[1, 2, 3]], dtype=np.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_transform_with_ops(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset, ops=[NumpyOpAdd1(inputs="x", outputs="y")])
        data = pipeline.transform(data=self.sample_data, mode="train")
        ans = {"x": np.array([[1, 2, 3]], dtype=np.float32), "y": np.array([[2, 3, 4]], dtype=np.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))


class TestPipelineGetResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_tf_dataset = get_sample_tf_dataset()
        cls.sample_torch_dataset = get_sample_torch_dataset()
        cls.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_get_result_tf_dataset_no_op(self):
        pipeline = fe.Pipeline(train_data=self.sample_tf_dataset)
        data = pipeline.get_results(num_steps=1)  # will ignore num_steps
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {
            "x": np.array([[0], [1], [2], [3]], dtype=np.float32),
            "y": np.array([[-99], [-98], [-97], [-96]], dtype=np.float32)
        }
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_torch_dataset_no_op(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)
        data = pipeline.get_results(num_steps=1)  # will not ignore num_steps
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {"x": np.array([0], dtype=np.float32), "y": np.array([-99], dtype=np.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_torch_dataloader_no_op(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataloader)
        data = pipeline.get_results(num_steps=1)  # will ignore num_steps
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {
            "x": np.array([[0], [1], [2], [3]], dtype=np.float32),
            "y": np.array([[-99], [-98], [-97], [-96]], dtype=np.float32)
        }
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_dict_batch_size(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset,
                               ops=NumpyOpAdd1(inputs="x", outputs="y"),
                               batch_size={"train": 1})
        data = pipeline.get_results(mode="train", epoch=1)
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {"x": np.array([[0]], dtype=np.float32), "y": np.array([[1]], dtype=np.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_dict_batch_size_scheduler(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset,
                               ops=NumpyOpAdd1(inputs="x", outputs="y"),
                               batch_size=EpochScheduler({1: {
                                   "train": 1
                               }}))
        data = pipeline.get_results(mode="train", epoch=1)
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {"x": np.array([[0]], dtype=np.float32), "y": np.array([[1]], dtype=np.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_dict_batch_size_train_eval(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset,
                               eval_data=self.sample_torch_dataset,
                               ops=NumpyOpAdd1(inputs="x", outputs="y"),
                               batch_size={
                                   "train": 2, "eval": 1
                               })
        data_train = pipeline.get_results(mode="train", epoch=1)
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        data_eval = pipeline.get_results(mode="eval", epoch=1)
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        data_train["x"] = data_train["x"].numpy()
        data_train["y"] = data_train["y"].numpy()
        data_eval["x"] = data_eval["x"].numpy()
        data_eval["y"] = data_eval["y"].numpy()
        ans_train = {"x": np.array([[0], [1]], dtype=np.float32), "y": np.array([[1], [2]], dtype=np.float32)}
        ans_eval = {"x": np.array([[0]], dtype=np.float32), "y": np.array([[1]], dtype=np.float32)}
        self.assertTrue(is_equal(data_train, ans_train))
        self.assertTrue(is_equal(data_eval, ans_eval))

    def test_pipeline_get_results_list_data(self):
        ds = self.sample_torch_dataset
        ds = ListData(ds)
        pipeline = fe.Pipeline(train_data=ds)
        data = pipeline.get_results()
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertEqual(data["x"].size(0), 5)

    def test_pipeline_get_results_batch_list_data(self):
        ds1 = self.sample_torch_dataset
        ds2 = ListData(ds1)
        batch_ds = BatchDataset(datasets=(ds1, ds2), num_samples=(2, 1))
        pipeline = fe.Pipeline(train_data=batch_ds)
        data = pipeline.get_results()
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertEqual(data["x"].size(0), 7)

    def test_pipeline_get_results_batch_list_data_disjoint_keys(self):
        ds1 = self.sample_torch_dataset
        ds2 = ListData(ds1, key1="x1", key2="y2")
        batch_ds = BatchDataset(datasets=(ds1, ds2), num_samples=(5, 1))
        pipeline = fe.Pipeline(train_data=batch_ds)
        data = pipeline.get_results()
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertEqual(data["x"].size(0), 5)


class TestPipelineGetLoader(unittest.TestCase):
    """ This test cover:
    * fe.pipeline.Pipeline.get_loader
    * fe.pipeline.Pipeline._pad_batch_collate


    This test has dependency on:
    * fe.schedule.schedule.get_current_items
    * fe.dataset.op_dataset.OpDataset
    * fe.pipeline.Pipeline._pad_batch_collate
    """
    @classmethod
    def setUpClass(cls):
        cls.sample_tf_dataset = get_sample_tf_dataset()
        cls.sample_torch_dataset = get_sample_torch_dataset()
        cls.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_get_loader_tf_dataset(self):
        pipeline = fe.Pipeline(train_data=self.sample_tf_dataset)
        loader = pipeline.get_loader(mode="train")
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertEqual(loader, self.sample_tf_dataset)

    def test_pipeline_get_loader_torch_dataloader(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataloader)
        loader = pipeline.get_loader(mode="train")
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertEqual(loader, self.sample_torch_dataloader)

    def test_pipeline_get_loader_torch_dataset(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)
        loader = pipeline.get_loader(mode="train", shuffle=False)
        with self.subTest("check loader type"):
            self.assertIsInstance(loader, torch.utils.data.DataLoader)
        with self.subTest("check data"):
            results = []
            for idx, batch in enumerate(loader, start=1):
                results.append(batch)
                if idx == 2:
                    break
            ans = [{
                "x": torch.tensor([0], dtype=torch.float32), "y": torch.tensor([-99], dtype=torch.float32)
            }, {
                "x": torch.tensor([1], dtype=torch.float32), "y": torch.tensor([-98], dtype=torch.float32)
            }]
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            self.assertTrue(is_equal(results, ans))

    def test_pipeline_get_loader_torch_dataset_with_batch_size(self):
        with self.subTest(shuffle=False):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            loader = pipeline.get_loader(mode="train", shuffle=False)
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            results = []
            for idx, batch in enumerate(loader, start=1):
                results.append(batch)
                if idx == 2:
                    break
            ans = [{
                "x": torch.tensor([[0], [1]], dtype=torch.float32),
                "y": torch.tensor([[-99], [-98]], dtype=torch.float32)
            },
                   {
                       "x": torch.tensor([[2], [3]], dtype=torch.float32),
                       "y": torch.tensor([[-97], [-96]], dtype=torch.float32)
                   }]
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            self.assertTrue(is_equal(results, ans))

        with self.subTest(shuffle=True):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            loader = pipeline.get_loader(mode="train", shuffle=True)
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            results = []
            for idx, batch in enumerate(loader, start=1):
                results.append(batch)
                if idx == 2:
                    break
            wrong_ans = [{
                "x": torch.tensor([[0], [1]], dtype=torch.float32),
                "y": torch.tensor([[-99], [-98]], dtype=torch.float32)
            },
                         {
                             "x": torch.tensor([[2], [3]], dtype=torch.float32),
                             "y": torch.tensor([[-97], [-96]], dtype=torch.float32)
                         }]
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            self.assertFalse(is_equal(results, wrong_ans))

        with self.subTest(shuffle=None):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            loader = pipeline.get_loader(mode="train", shuffle=None)
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            results = []
            for idx, batch in enumerate(loader, start=1):
                results.append(batch)
                if idx == 2:
                    break
            wrong_ans = [{
                "x": torch.tensor([[0], [1]], dtype=torch.float32),
                "y": torch.tensor([[-99], [-98]], dtype=torch.float32)
            },
                         {
                             "x": torch.tensor([[2], [3]], dtype=torch.float32),
                             "y": torch.tensor([[-97], [-96]], dtype=torch.float32)
                         }]
            time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
            self.assertFalse(is_equal(results,
                                      wrong_ans))  # if shuffle is None and has specify batch_size, it will shuffle

    def test_pipeline_get_loader_torch_dataset_pad(self):
        """
        [[1],    =>  [[1, -1],
         [1]]         [1, -1]]

        [[1, 1]] =>  [[1, 1],
                      [-1, -1]]
        """
        dataset = fe.dataset.NumpyDataset({"x": [np.ones((2, 1), dtype=np.float32), np.ones((1, 2), dtype=np.float32)]})
        pipeline = fe.Pipeline(train_data=dataset, pad_value=-1, batch_size=2)
        loader = pipeline.get_loader(mode="train", shuffle=False)
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        for idx, batch in enumerate(loader, start=1):
            result = batch
            if idx == 1:
                break
        ans = {"x": torch.tensor([[[1, -1], [1, -1]], [[1, 1], [-1, -1]]], dtype=torch.float32)}
        time.sleep(2)  # avoid  DataLoader worker (pid #) is killed by signal: Aborted.
        self.assertTrue(is_equal(ans, result))
