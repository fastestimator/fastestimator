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
import itertools
import unittest

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe
from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.dataset.extend_dataset import ExtendDataset
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.numpyop import NumpyOp, RemoveIf
from fastestimator.op.numpyop.univariate import Minmax
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
    * fe.schedule.schedule.EpochScheduler
    """
    def setUp(self):
        self.sample_tf_dataset = get_sample_tf_dataset()
        self.sample_torch_dataset = get_sample_torch_dataset()
        self.sample_torch_dataloader = get_sample_torch_dataloader()
        self.sample_numpy_op = SampleNumpyOp(inputs="x", outputs="x")
        self.sample_tensor_op = SampleTensorOp(inputs="x", outputs="x")

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
                self.fail("exception occurred")

        with self.subTest("with batch_size not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, batch_size=10)
            except:
                self.fail("exception occurred")

        with self.subTest("with num_process not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, num_process=1)
            except:
                self.fail("exception occurred")

    def test_pipeline_init_torch_dataset_scheduler_have_op_batch_size_num_process(self):
        data = EpochScheduler(epoch_dict={1: self.sample_torch_dataset, 2: None})

        with self.subTest("with numpyop"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, ops=[self.sample_numpy_op])
            except:
                self.fail("exception occurred")

        with self.subTest("with batch_size not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, batch_size=10)
            except:
                self.fail("exception occurred")

        with self.subTest("with num_process not None"):
            try:
                pipeline = fe.Pipeline(train_data=data, eval_data=data, test_data=data, num_process=1)
            except:
                self.fail("exception occurred")

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
                    self.fail("exception occurred")

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
    * fe.schedule.schedule.EpochScheduler
    """
    def setUp(self):
        self.sample_torch_dataset = get_sample_torch_dataset()

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
    * fe.schedule.schedule.EpochScheduler
    """
    def setUp(self):
        self.sample_torch_dataset = get_sample_torch_dataset()

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


class TestPipelineBenchmark(unittest.TestCase):
    """ This test has dependency on:
    * fe.pipeline.Pipeline.get_loader
    """
    def setUp(self):
        self.sample_tf_dataset = get_sample_tf_dataset()
        self.sample_torch_dataset = get_sample_torch_dataset()
        self.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_benchmark_smoke(self):
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
                    self.fail("exception occurred")

    def test_multi_ds(self):
        ds = {"ds_1": self.sample_torch_dataset, "ds_2": self.sample_torch_dataset}
        pipeline = fe.Pipeline(train_data=ds)
        pipeline.benchmark()

    def test_multi_eval(self):
        ds = {"ds_1": self.sample_torch_dataset, "ds_2": self.sample_torch_dataset}
        pipeline = fe.Pipeline(eval_data=ds)
        pipeline.benchmark(mode="eval")

    def test_multi_test(self):
        ds = {"ds_1": self.sample_torch_dataset, "ds_2": self.sample_torch_dataset}
        pipeline = fe.Pipeline(test_data=ds)
        pipeline.benchmark(mode="test")


class TestPipelineTransform(unittest.TestCase):
    """ This test has dependency on:
    * fe.schedule.schedule.get_current_items
    * fe.op.numpy.numpy.forward_numpy
    """
    def setUp(self):
        self.sample_data = {"x": np.array([1, 2, 3], dtype=np.float32)}
        self.sample_dataset = get_sample_torch_dataset()

    def test_pipeline_transform_no_ops(self):
        pipeline = fe.Pipeline()
        data = pipeline.transform(data=self.sample_data, mode="train")
        ans = {"x": np.array([[1, 2, 3]], dtype=np.float32)}
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_transform_with_ops(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset, ops=[NumpyOpAdd1(inputs="x", outputs="y")])
        data = pipeline.transform(data=self.sample_data, mode="train")
        ans = {"x": np.array([[1, 2, 3]], dtype=np.float32), "y": np.array([[2, 3, 4]], dtype=np.float32)}
        self.assertTrue(is_equal(data, ans))

    def test_multi_train(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset,
                               ops=Minmax(inputs="x", outputs="x", ds_id=("ds_1", "ds_2")))
        sample_data = {"x": np.array([0, 255], dtype=np.float32)}
        data1 = pipeline.transform(data=sample_data, mode="train", ds_id="ds_1")
        assert data1["x"].max() == 1.0
        data2 = pipeline.transform(data=sample_data, mode="train", ds_id="ds_2")
        assert data2["x"].max() == 1.0
        data3 = pipeline.transform(data=sample_data, mode="train", ds_id="ds_3")
        assert data3["x"].max() == 255

    def test_multi_eval(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset,
                               ops=Minmax(inputs="x", outputs="x", ds_id=("!ds_1", "!ds_2")))
        sample_data = {"x": np.array([0, 255], dtype=np.float32)}
        data1 = pipeline.transform(data=sample_data, mode="eval", ds_id="ds_1")
        assert data1["x"].max() == 255
        data2 = pipeline.transform(data=sample_data, mode="eval", ds_id="ds_2")
        assert data2["x"].max() == 255
        data3 = pipeline.transform(data=sample_data, mode="eval", ds_id="ds_3")
        assert data3["x"].max() == 1.0

    def test_multi_test(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset, ops=Minmax(inputs="x", outputs="x", ds_id="!ds_1"))
        sample_data = {"x": np.array([0, 255], dtype=np.float32)}
        data1 = pipeline.transform(data=sample_data, mode="test", ds_id="ds_1")
        assert data1["x"].max() == 255
        data2 = pipeline.transform(data=sample_data, mode="test", ds_id="ds_2")
        assert data2["x"].max() == 1.0
        data3 = pipeline.transform(data=sample_data, mode="test", ds_id="ds_3")
        assert data3["x"].max() == 1.0

    def test_multi_infer(self):
        pipeline = fe.Pipeline(train_data=self.sample_dataset, ops=[Minmax(inputs="x", outputs="x", ds_id="!ds_1")])
        sample_data = {"x": np.array([0, 255], dtype=np.float32)}
        data1 = pipeline.transform(data=sample_data, mode="infer", ds_id="ds_1")
        assert data1["x"].max() == 255
        data2 = pipeline.transform(data=sample_data, mode="infer", ds_id="ds_2")
        assert data2["x"].max() == 1.0
        data3 = pipeline.transform(data=sample_data, mode="infer", ds_id="ds_3")
        assert data3["x"].max() == 1.0
        data4 = pipeline.transform(data=sample_data, mode="infer")
        assert data4["x"].max() == 1.0


class TestPipelineGetResults(unittest.TestCase):
    def setUp(self):
        self.sample_tf_dataset = get_sample_tf_dataset()
        self.sample_torch_dataset = get_sample_torch_dataset()
        self.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_get_result_tf_dataset_no_op(self):
        pipeline = fe.Pipeline(train_data=self.sample_tf_dataset)
        data = pipeline.get_results(num_steps=1)  # will ignore num_steps
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {
            "x": np.array([[0], [1], [2], [3]], dtype=np.float32),
            "y": np.array([[-99], [-98], [-97], [-96]], dtype=np.float32)
        }
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_torch_dataset_no_op(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)
        data = pipeline.get_results(num_steps=1)  # will not ignore num_steps
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {"x": np.array([0], dtype=np.float32), "y": np.array([-99], dtype=np.float32)}
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
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_dict_batch_size(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset,
                               ops=NumpyOpAdd1(inputs="x", outputs="y"),
                               batch_size={"train": 1})
        data = pipeline.get_results(mode="train", epoch=1)
        data["x"] = data["x"].numpy()
        data["y"] = data["y"].numpy()
        ans = {"x": np.array([[0]], dtype=np.float32), "y": np.array([[1]], dtype=np.float32)}
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
        self.assertTrue(is_equal(data, ans))

    def test_pipeline_get_result_dict_batch_size_train_eval(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset,
                               eval_data=self.sample_torch_dataset,
                               ops=NumpyOpAdd1(inputs="x", outputs="y"),
                               batch_size={
                                   "train": 2, "eval": 1
                               })
        data_train = pipeline.get_results(mode="train", epoch=1)
        data_eval = pipeline.get_results(mode="eval", epoch=1)
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
        self.assertEqual(data["x"].size(0), 5)

    def test_pipeline_get_results_batch_list_data(self):
        ds1 = self.sample_torch_dataset
        ds2 = ListData(ds1)
        batch_ds = BatchDataset(datasets=(ds1, ds2), num_samples=(2, 1))
        pipeline = fe.Pipeline(train_data=batch_ds)
        data = pipeline.get_results()
        self.assertEqual(data["x"].size(0), 7)

    def test_pipeline_get_results_batch_list_data_disjoint_keys(self):
        ds1 = self.sample_torch_dataset
        ds2 = ListData(ds1, key1="x1", key2="y2")
        batch_ds = BatchDataset(datasets=(ds1, ds2), num_samples=(5, 1))
        pipeline = fe.Pipeline(train_data=batch_ds)
        data = pipeline.get_results()
        self.assertEqual(data["x"].size(0), 5)

    def test_multi_train(self):
        train_data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"ds_1": train_data, "ds_2": train_data, "ds_3": train_data}
        pipeline = fe.Pipeline(train_data=train_ds,
                               batch_size=1,
                               ops=Minmax(inputs="x", outputs="x", ds_id=("ds_1", "ds_2")))
        data1 = pipeline.get_results(mode="train", ds_id="ds_3")
        assert data1["x"].numpy().max() == 255
        data2 = pipeline.get_results(mode="train", ds_id="ds_1")
        assert data2["x"].numpy().max() == 1.0
        data3 = pipeline.get_results(mode="train", ds_id="ds_2")
        assert data3["x"].numpy().max() == 1.0

    def test_multi_eval(self):
        eval_data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        eval_ds = {"ds_1": eval_data, "ds_2": eval_data, "ds_3": eval_data}
        pipeline = fe.Pipeline(eval_data=eval_ds,
                               batch_size=1,
                               ops=Minmax(inputs="x", outputs="x", ds_id=("!ds_1", "!ds_2")))
        data1 = pipeline.get_results(mode="eval", ds_id="ds_3")
        assert data1["x"].numpy().max() == 1.0
        data2 = pipeline.get_results(mode="eval", ds_id="ds_1")
        assert data2["x"].numpy().max() == 255
        data3 = pipeline.get_results(mode="eval", ds_id="ds_2")
        assert data3["x"].numpy().max() == 255

    def test_multi_test(self):
        test_data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        test_ds = {"ds_1": test_data, "ds_2": test_data, "ds_3": test_data}
        pipeline = fe.Pipeline(test_data=test_ds, batch_size=1, ops=Minmax(inputs="x", outputs="x", ds_id="!ds_1"))
        data1 = pipeline.get_results(mode="test", ds_id="ds_3")
        assert data1["x"].numpy().max() == 1.0
        data2 = pipeline.get_results(mode="test", ds_id="ds_1")
        assert data2["x"].numpy().max() == 255
        data3 = pipeline.get_results(mode="test", ds_id="ds_2")
        assert data3["x"].numpy().max() == 1.0

    def test_multi_train_scheduler(self):
        train_ds = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds2 = NumpyDataset({"x": np.array([[0, 256], [256, 0]])})
        train_data1 = train_ds
        train_data2 = EpochScheduler({1: train_ds, 2: None})
        train_data3 = EpochScheduler({1: train_ds, 2: train_ds2})
        train_dataset_overall = {"ds_1": train_data1, "ds_2": train_data2, "ds_3": train_data3}
        pipeline = fe.Pipeline(train_data=train_dataset_overall,
                               batch_size=1,
                               ops=Minmax(inputs="x", outputs="x", ds_id=("ds_1", "ds_2")))
        data1 = pipeline.get_results(mode="train", ds_id="ds_3", epoch=1)
        assert data1["x"].numpy().max() == 255
        data2 = pipeline.get_results(mode="train", ds_id="ds_3", epoch=2)
        assert data2["x"].numpy().max() == 256
        data3 = pipeline.get_results(mode="train", ds_id="ds_1", epoch=1)
        assert data3["x"].numpy().max() == 1.0
        data4 = pipeline.get_results(mode="train", ds_id="ds_2", epoch=2)
        assert data4 == []


class TestPipelineGetLoader(unittest.TestCase):
    """ This test cover:
    * fe.pipeline.Pipeline.get_loader
    * fe.pipeline.Pipeline._pad_batch_collate


    This test has dependency on:
    * fe.schedule.schedule.get_current_items
    * fe.dataset.op_dataset.OpDataset
    * fe.pipeline.Pipeline._pad_batch_collate
    """
    def setUp(self):
        self.sample_tf_dataset = get_sample_tf_dataset()
        self.sample_torch_dataset = get_sample_torch_dataset()
        self.sample_torch_dataloader = get_sample_torch_dataloader()

    def test_pipeline_get_loader_tf_dataset(self):
        pipeline = fe.Pipeline(train_data=self.sample_tf_dataset)
        with pipeline(mode="train") as loader:
            self.assertEqual(loader, self.sample_tf_dataset)

    def test_pipeline_get_loader_torch_dataloader(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataloader)
        with pipeline(mode="train") as loader:
            self.assertEqual(loader, self.sample_torch_dataloader)

    def test_pipeline_get_loader_torch_dataset(self):
        pipeline = fe.Pipeline(train_data=self.sample_torch_dataset)
        with pipeline(mode="train", shuffle=False) as loader:
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
                self.assertTrue(is_equal(results, ans))

    def test_pipeline_get_loader_torch_dataset_with_batch_size(self):
        with self.subTest(shuffle=False):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            with pipeline(mode="train", shuffle=False) as loader:
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
                self.assertTrue(is_equal(results, ans))

        with self.subTest(shuffle=True):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            with pipeline(mode="train", shuffle=True) as loader:
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
                self.assertFalse(is_equal(results, wrong_ans))

        with self.subTest(shuffle=None):
            pipeline = fe.Pipeline(train_data=self.sample_torch_dataset, batch_size=2)
            with pipeline(mode="train", shuffle=None) as loader:
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
        with pipeline(mode="train", shuffle=False) as loader:
            for idx, batch in enumerate(loader, start=1):
                result = batch
                if idx == 1:
                    break
            ans = {"x": torch.tensor([[[1, -1], [1, -1]], [[1, 1], [-1, -1]]], dtype=torch.float32)}
            self.assertTrue(is_equal(ans, result))

    def test_pipeline_nested_loaders(self):
        dataset = fe.dataset.NumpyDataset({"x": [np.ones((2, 1), dtype=np.float32), np.ones((1, 2), dtype=np.float32)]})
        pipeline = fe.Pipeline(train_data=dataset, pad_value=-1, batch_size=2)
        with self.subTest("With Call"):
            with self.assertRaises(ValueError):
                with pipeline(mode='train') as loader1:
                    with pipeline(mode='eval') as loader2:
                        print(loader1)
                        print(loader2)
        with self.subTest("Without Call"):
            with self.assertRaises(ValueError):
                with pipeline(mode='train') as loader1:
                    with pipeline as loader2:
                        print(loader1)
                        print(loader2)

    def test_pipeline_nested_call(self):
        dataset = fe.dataset.NumpyDataset({"x": [np.ones((2, 1), dtype=np.float32), np.ones((1, 2), dtype=np.float32)]})
        pipeline = fe.Pipeline(train_data=dataset, pad_value=-1, batch_size=2)
        with self.assertRaises(ValueError):
            with pipeline(mode='train') as loader1:
                pipeline(mode='train')
                print(loader1)


class TestPipelineNames(unittest.TestCase):
    def test_forbidden_names_none(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {None: data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)

    def test_forbidden_names_esc(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"!ds1": data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)

    def test_forbidden_names_semi(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"ds1;": data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)

    def test_forbidden_names_colon(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"ds1:": data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)

    def test_forbidden_names_empty(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"": data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)

    def test_forbidden_names_pipe(self):
        data = NumpyDataset({"x": np.array([[0, 255], [255, 0]])})
        train_ds = {"ds|ds1": data}
        with self.assertRaises(AssertionError):
            fe.Pipeline(train_data=train_ds)


class TestPipelineFilter(unittest.TestCase):

    def test_unbatched_nodrop_nofilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(23)], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertSetEqual(set([i for i in range(23)]), set(composite_list))

    def test_unbatched_nodrop_replacementfilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                target = [0, 1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 3, 4, 5]
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(target, composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(23, len(composite_list))
                    self.assertSetEqual(set(target), set(composite_list))  # Should visit all the data at least once

    def test_unbatched_nodrop_cutfilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [2, 6, 9, 10, 11],
                                                    inputs="idx",
                                                    replacement=False))
                target = [0, 1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(target, composite_list)

                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(18, len(composite_list))
                    self.assertSetEqual(set(target), set(composite_list))  # Should visit all the data at least once

    def test_unbatched_drop_nofilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       drop_last=True)
                with self.subTest("shuffle false"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(20)], composite_list)
                with self.subTest("shuffle true"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(20, len(composite_list))  # Since shuffling don't know which will be kept

    def test_unbatched_drop_replacementfilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       drop_last=True,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                target = [0, 1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1]
                with self.subTest("shuffle false"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(target, composite_list)
                with self.subTest("shuffle true"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(20, len(composite_list))  # Since shuffling don't know which will be kept
                    self.assertSetEqual(set(target), set(composite_list))  # Everything should be visited at least once

    def test_unbatched_drop_cutfilter(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       drop_last=True,
                                       ops=RemoveIf(fn=lambda x: x in [2, 6, 9, 10, 11],
                                                    inputs="idx",
                                                    replacement=False))
                target = [0, 1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19]
                with self.subTest("shuffle false"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(target, composite_list)
                with self.subTest("shuffle true"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(15, len(composite_list))
                    # Don't know which ones, but should be no repeats at least
                    self.assertEqual(15, len(set(composite_list)))

    def test_unbatched_nofilter_expand(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(23)]*2+[i for i in range(9)], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_unbatched_nofilter_contract(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(10)], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_unbatched_replacementfilter_expand(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(([0, 1, 3, 4, 5, 7, 8]+list(range(12, 23)))*3+[0], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in [0, 1, 3, 4, 5, 7, 8]+list(range(12, 23)):
                        self.assertGreaterEqual(composite_list.count(i), 3)

    def test_unbatched_replacementfilter_contract(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    # The initial data will be deferred until later since it only formed a partial batch
                    self.assertListEqual([15, 16, 17, 18, 19, 0, 8, 9, 14, 20], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_unbatched_cutfilter_expand(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx",
                                                    replacement=False,
                                                    fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual((([0, 1, 3, 4, 5, 7, 8]+list(range(12, 23)))*3)[:43], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 40)
                    self.assertLessEqual(len(composite_list), 45)
                    for i in [0, 1, 3, 4, 5, 7, 8]+list(range(12, 23)):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_unbatched_cutfilter_contract(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10],
                                                    replacement=False,
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([0, 8, 9], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(set(composite_list)), 2)

    def test_unbatched_nofilter_expandds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=55)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(23)] * 2 + [i for i in range(9)], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_unbatched_nofilter_contractds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=10)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([i for i in range(10)], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_unbatched_replacementfilter_expandds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=55)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual(([0, 1, 3, 4, 5, 7, 8] + list(range(12, 23))) * 3 + [0], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in [0, 1, 3, 4, 5, 7, 8] + list(range(12, 23)):
                        self.assertGreaterEqual(composite_list.count(i), 3)

    def test_unbatched_replacementfilter_contractds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=10)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    # Batch order will be modified since initial data is deferred due to filter
                    self.assertListEqual([15, 16, 17, 18, 19, 0, 8, 9, 14, 20], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_unbatched_cutfilter_expandds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=55)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx",
                                                    replacement=False,
                                                    fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual((([0, 1, 3, 4, 5, 7, 8] + list(range(12, 23))) * 3)[:43], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 40)
                    self.assertLessEqual(len(composite_list), 45)
                    for i in [0, 1, 3, 4, 5, 7, 8] + list(range(12, 23)):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_unbatched_cutfilter_contractds(self):
        data = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data = ExtendDataset(data, spoof_length=10)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10],
                                                    replacement=False,
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertListEqual([0, 8, 9], composite_list)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(set(composite_list)), 2)

    # ### Batched Tests ### #
    # Note that pipeline ignores drop_last for batched datasets
    # ###               ### #

    def test_batched_nofilter(self):
        data_a = NumpyDataset({"a": np.array([i for i in range(23)])})
        data_b = NumpyDataset({"b": np.array([i for i in range(23)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                with self.subTest("shuffle False"):
                    data = BatchDataset(datasets=[data_a, data_b], num_samples=[3, 3])
                    pipeline = fe.Pipeline(train_data=data,
                                           batch_size=5,
                                           num_process=n_process)
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertEqual(24, len(composite_a))  # batch dataset will fill up the final batch
                    self.assertListEqual(composite_a, composite_b)  # make sure that the index orders are consistent
                    self.assertEqual(23, len(set(composite_a)))  # make sure all the elements got visited
                with self.subTest("shuffle True"):
                    # have to re-create the dataset since shuffle pollutes the state of the index maps
                    data = BatchDataset(datasets=[data_a, data_b], num_samples=[3, 3])
                    pipeline = fe.Pipeline(train_data=data,
                                           batch_size=5,
                                           num_process=n_process)
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertEqual(24, len(composite_a))  # batch dataset will fill up the final batch
                    self.assertEqual(24, len(composite_b))  # batch dataset will fill up the final batch
                    self.assertNotEqual(composite_a, composite_b)  # make sure that the index orders are mixed
                    self.assertEqual(23, len(set(composite_a)))  # make sure all the elements got visited
                    self.assertEqual(23, len(set(composite_b)))  # make sure all the elements got visited

    def test_batched_replacementfilter(self):
        data_a = NumpyDataset({"a": np.array([i for i in range(29)])})
        data_b = NumpyDataset({"b": np.array([i for i in range(29)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                data = BatchDataset(datasets=[data_a, data_b], num_samples=[3, 3])
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="a", fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertEqual(30, len(composite_a))  # batch dataset will fill up the final batch
                    self.assertListEqual(composite_a, composite_b)  # make sure that the index orders are consistent
                    self.assertGreaterEqual(len(set(composite_a)), 12)  # There should be at least 4 unique batches
                with self.subTest("shuffle True"):
                    # have to re-create the dataset since shuffle pollutes the state of the index maps
                    data = BatchDataset(datasets=[data_a, data_b], num_samples=[5, 5])
                    pipeline = fe.Pipeline(train_data=data,
                                           batch_size=5,
                                           num_process=n_process,
                                           ops=RemoveIf(inputs="a", fn=lambda x: x in [2, 6, 9, 10]))
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertEqual(30, len(composite_a))  # batch dataset will fill up the final batch
                    self.assertEqual(30, len(composite_b))  # batch dataset will fill up the final batch
                    self.assertNotEqual(composite_a, composite_b)  # make sure that the index orders are mixed
                    batch_pairs = []
                    for batch in batches:
                        batch_a = list(batch['a'].numpy())
                        batch_b = list(batch['b'].numpy())
                        batch_pairs.append(sorted([(a, b) for a, b in zip(batch_a, batch_b)], key=lambda t: t[0]))
                    for pair1, pair2 in itertools.combinations(batch_pairs, 2):
                        # Make sure none of the batch sequences are ever repeated
                        self.assertNotEqual(pair1, pair2)

    def test_batched_cutfilter(self):
        data_a = NumpyDataset({"a": np.array([i for i in range(29)])})
        data_b = NumpyDataset({"b": np.array([i for i in range(29)])})
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                data = BatchDataset(datasets=[data_a, data_b], num_samples=[3, 3])
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="a",
                                                    replacement=False,
                                                    fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertListEqual(composite_a, composite_b)  # make sure that the index orders are consistent
                    self.assertGreaterEqual(len(set(composite_a)), 12)  # There should be at least 4 unique batches
                    self.assertGreaterEqual(len(composite_a), 15)  # There should be at least 5 batches
                with self.subTest("shuffle True"):
                    # have to re-create the dataset since shuffle pollutes the state of the index maps
                    data = BatchDataset(datasets=[data_a, data_b], num_samples=[5, 5])
                    pipeline = fe.Pipeline(train_data=data,
                                           batch_size=5,
                                           num_process=n_process,
                                           ops=RemoveIf(inputs="a",
                                                        replacement=False,
                                                        fn=lambda x: x in [2, 6, 9]))
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_a = list(np.concatenate([batch['a'] for batch in batches]))
                    composite_b = list(np.concatenate([batch['b'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_a), 10)  # There should be at least 2 batches
                    self.assertNotEqual(composite_a, composite_b)  # make sure that the index orders are mixed
                    batch_pairs = []
                    for batch in batches:
                        batch_a = list(batch['a'].numpy())
                        batch_b = list(batch['b'].numpy())
                        batch_pairs.append(sorted([(a, b) for a, b in zip(batch_a, batch_b)], key=lambda t: t[0]))
                    for pair1, pair2 in itertools.combinations(batch_pairs, 2):
                        # Make sure none of the batch sequences are ever repeated
                        self.assertNotEqual(pair1, pair2)

    def test_batched_nofilter_expand(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_batched_nofilter_contract(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(25)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(25)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_batched_replacementfilter_expand(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))

    def test_batched_replacementfilter_contract(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(composite_list))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(composite_list))

    def test_batched_cutfilter_expand(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx",
                                                    replacement=False,
                                                    fn=lambda x: x in [2, 6, 9]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 10)
                    self.assertLessEqual(len(composite_list), 45)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=11) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 10)
                    self.assertLessEqual(len(composite_list), 45)

    def test_batched_cutfilter_contract(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in list(range(18)),
                                                    replacement=False,
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    self.assertEqual(0, len(batches))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True, steps_per_epoch=2) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    self.assertEqual(0, len(batches))

    def test_batched_nofilter_expandds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=11)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                    for i in range(23):
                        self.assertGreaterEqual(composite_list.count(i), 2)

    def test_batched_nofilter_contractds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(25)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(25)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=2)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process)
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(set(composite_list)))  # All the elements should be unique

    def test_batched_replacementfilter_expandds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=11)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx", fn=lambda x: x in [2, 6, 9, 10, 11]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(55, len(composite_list))

    def test_batched_replacementfilter_contractds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=2)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(composite_list))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertEqual(10, len(composite_list))

    def test_batched_cutfilter_expandds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=11)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(inputs="idx",
                                                    replacement=False,
                                                    fn=lambda x: x in [2, 6, 9]))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 10)
                    self.assertLessEqual(len(composite_list), 45)
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    composite_list = list(np.concatenate([batch['idx'] for batch in batches]))
                    self.assertGreaterEqual(len(composite_list), 10)
                    self.assertLessEqual(len(composite_list), 45)

    def test_batched_cutfilter_contractds(self):
        data_x = NumpyDataset({"idx": np.array([i for i in range(23)])})
        data_y = NumpyDataset({"y": np.array([i for i in range(23)])})
        data = BatchDataset(datasets=[data_x, data_y], num_samples=[5, 5])
        data = ExtendDataset(data, spoof_length=2)
        for n_process in [0, 7]:
            with self.subTest("proc status", workers=n_process):
                pipeline = fe.Pipeline(train_data=data,
                                       batch_size=5,
                                       num_process=n_process,
                                       ops=RemoveIf(fn=lambda x: x in list(range(18)),
                                                    replacement=False,
                                                    inputs="idx"))
                with self.subTest("shuffle False"):
                    with pipeline(mode="train", shuffle=False) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    self.assertEqual(0, len(batches))
                with self.subTest("shuffle True"):
                    with pipeline(mode="train", shuffle=True) as loader:
                        itr = iter(loader)
                        batches = []
                        for elem in itr:
                            batches.append(elem)
                    self.assertEqual(0, len(batches))
