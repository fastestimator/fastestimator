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
import json
import multiprocessing as mp
import os
import time

import numpy as np
import tensorflow as tf

from fastestimator.pipeline.processing import TensorFilter
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import flatten_operation, get_op_from_mode, verify_ops
from fastestimator.util.tfrecord import get_features
from fastestimator.util.util import convert_tf_dtype
from fastestimator.util.schedule import Scheduler


class Pipeline:
    def __init__(self, batch_size, data=None, ops=None, read_feature=None, padded_batch=False, expand_dims=False,
                 max_shuffle_buffer_mb=3000):

        self.batch_size = batch_size
        self.data = data
        self.ops = ops
        self.read_feature = read_feature
        self.padded_batch = padded_batch
        self.expand_dims = expand_dims
        self.max_shuffle_buffer_mb = max_shuffle_buffer_mb
        self.mode_forward_ops = {"train": [], "eval": []}
        self.mode_filter_ops = {"train": [], "eval": []}
        self.feature_name = {"train": [], "eval": []}
        self.num_examples = {"train": [], "eval": []}
        self.all_features = {"train": [], "eval": []}
        self.shuffle_buffer = {"train": [], "eval": []}
        self.padded_feature_shape = {"train": {}, "eval": {}}
        self.num_core = mp.cpu_count()
        self._verify_input()

    def _verify_input(self):
        if self.data:
            assert isinstance(self.data, (dict, tuple, RecordWriter)), "data must be either RecordWriter, dictionary"
        if self.read_feature:
            assert isinstance(self.read_feature, (list, dict, tuple)), "write_feature must be either list, dictionary"
        if any(isinstance(inp, tuple) for inp in [self.data, self.read_feature]):
            num_unpaired_feature_sets = []
            if self.data and not isinstance(self.data, RecordWriter):
                assert isinstance(self.data, tuple), "data must be in tuple format when using unpaired feature set"
                num_unpaired_feature_sets.append(len(self.data))
            if self.read_feature:
                assert isinstance(self.read_feature,
                                  tuple), "read feature must be in tuple format when using unpaired feature set"
                num_unpaired_feature_sets.append(len(self.read_feature))
            assert len(set(
                num_unpaired_feature_sets)) == 1, "tuple length should be consistent when creating unpaired feature set"
        else:
            if self.read_feature:
                self.read_feature = [self.read_feature]
        if self.ops:
            self.ops = flatten_operation(self.ops)

    def _prepare(self, inputs):
        self.inputs = inputs
        if isinstance(self.data, (tuple, dict)):
            if isinstance(self.data, dict):
                self.data = [self.data]
            else:
                self.data = list(self.data)
            for data in self.data:
                for key in data.keys():
                    self.all_features[key].append(data[key])
        else:
            assert inputs, "Must specify the data path of tfrecords"
            if isinstance(self.data, RecordWriter) and (not os.path.exists(inputs) or len(os.listdir(inputs)) == 0):
                self.data.create_tfrecord(save_dir=inputs)
            else:
                print("FastEstimator: Using existing tfrecords in {}".format(inputs))
            self.feature_dtype = {"train": [], "eval": []}
            self.record_feature_shape = {"train": [], "eval": []}
            self.compression = {"train": [], "eval": []}
            self.file_names = {"train": [], "eval": []}

    def _prepare_mode(self, mode):
        success = True
        if isinstance(self.data, list):
            self._get_numpy_config(mode)
        else:
            success = self._get_tfrecord_config(mode)
        if success:
            self._get_feature_name(mode=mode)
            self._check_ops(mode=mode)
            if self.padded_batch:
                _ = self._input_stream(mode, warm_up=True)
        return success

    def _get_numpy_config(self, mode):
        for data in self.all_features[mode]:
            num_example_list = []
            for key in data.keys():
                feature_data = data[key]
                if type(feature_data) is list:
                    num_example_list.append(len(feature_data))
                elif type(feature_data) is np.ndarray:
                    num_example_list.append(feature_data.shape[0])
                else:
                    raise ValueError("the feature only supports list or numpy array")
            assert len(set(num_example_list)
                       ) == 1, "inconsistent number of data found during {}, please check the data".format(mode)
            self.num_examples[mode].append(set(num_example_list).pop())
            self.shuffle_buffer[mode].append(set(num_example_list).pop())

    def _get_feature_name(self, mode):
        if len(self.all_features[mode]) > 1 and self.read_feature:
            assert isinstance(self.read_feature, tuple), "read feature must be a tuple for unpaired feature set"
            assert len(self.read_feature) == len(
                self.all_features[mode]), "the tuple should be consistent between read feature and data"
        for idx, feature in enumerate(self.all_features[mode]):
            if self.read_feature is None:
                self.feature_name[mode].append(list(feature.keys()))
            elif isinstance(self.read_feature[idx], dict):
                self.feature_name[mode].append(self.read_feature[idx][mode])
            else:
                self.feature_name[mode].append(self.read_feature[idx])

    def _check_ops(self, mode):
        if self.ops:
            mode_ops = get_op_from_mode(self.ops, mode)
            mode_ops_without_filter = [op for op in mode_ops if not isinstance(op, TensorFilter)]
            if len(mode_ops_without_filter) > 0:
                verify_ops(mode_ops_without_filter, "Pipeline")
                forward_ops = []
                for op in mode_ops:
                    if not isinstance(op, TensorFilter):
                        forward_ops.append(op)
                    else:
                        self.mode_forward_ops[mode].append(forward_ops)
                        self.mode_filter_ops[mode].append(op)
                        forward_ops = []
                self.mode_forward_ops[mode].append(forward_ops)

    def _get_tfrecord_config(self, mode):
        json_files = [
            os.path.join(self.inputs, f) for f in os.listdir(self.inputs)
            if f.endswith(".json") and f.startswith("%s_summary" % mode)
        ]
        found_record = len(json_files) > 0
        if found_record:
            for json_file in json_files:
                with open(json_file, 'r') as fp:
                    summary = json.load(fp)
                self.file_names[mode].append([os.path.join(self.inputs, f) for f in summary["file_names"]])
                self.num_examples[mode].append(np.sum(summary["num_examples"]))
                self.feature_dtype[mode].append(summary["feature_dtype"])
                self.record_feature_shape[mode].append(summary["feature_shape"])
                if "compression" in summary:
                    self.compression[mode].append(summary["compression"])
                else:
                    self.compression[mode].append(None)
                self.all_features[mode].append(
                    get_features(self.file_names[mode][-1][0], compression=self.compression[mode][-1]))
                num_example = self.num_examples[mode][-1]
                example_size_mb = summary["example_size_mb"]
                self.shuffle_buffer[mode].append(int(min(num_example, self.max_shuffle_buffer_mb // example_size_mb)))
                print("FastEstimator: Found %d examples for %s in %s" % (num_example, mode, json_file))
        return found_record

    def _input_stream(self, state, warm_up=False):
        mode = state["mode"]
        ds_tuple = ()
        # Data Reading
        for idx in range(len(self.all_features[mode])):
            if isinstance(self.data, (list, tuple)):
                ds = tf.data.Dataset.from_tensor_slices(self.all_features[mode][idx])
            else:
                if mode == "train":
                    ds = tf.data.Dataset.from_tensor_slices(self.file_names[mode][idx])
                    ds = ds.shuffle(len(self.file_names[mode][idx]))
                    ds = ds.interleave(
                        lambda ds: tf.data.TFRecordDataset(ds, compression_type=self.compression[mode][idx]),
                        cycle_length=self.num_core, block_length=2)
                else:
                    ds = tf.data.TFRecordDataset(self.file_names[mode][idx],
                                                 compression_type=self.compression[mode][idx])
                ds = ds.map(lambda ds: self._decode_records(ds, mode, idx), num_parallel_calls=self.num_core)
            if mode == "train":
                ds = ds.shuffle(self.shuffle_buffer[mode][idx])
                ds = ds.repeat()
            ds_tuple += ds,
        # Combine dataset from different unpaired feature sets
        if len(self.all_features[mode]) > 1:
            dataset = tf.data.Dataset.zip(ds_tuple)
            dataset = dataset.map(self._combine_dataset, num_parallel_calls=self.num_core)
        else:
            dataset = ds_tuple[0]
        # Data preprocessing
        if self.ops:
            dataset = self._execute_mode_ops(dataset, mode)
        if self.expand_dims:
            dataset = dataset.flat_map(lambda dataset: tf.data.Dataset.from_tensor_slices(dataset))
        if warm_up:
            dataset = dataset.map(lambda dataset: self._get_padded_shape(dataset, mode))
        # Decide batch size
        if isinstance(self.batch_size, Scheduler) and self.batch_size.change_value(state):
            self.batch_size_tensor = tf.convert_to_tensor(self.batch_size.get_current_value(state), dtype=tf.int64)
        else:
            self.batch_size_tensor = tf.convert_to_tensor(self.batch_size, dtype=tf.int64)
        if self.padded_batch:
            dataset = dataset.padded_batch(self.batch_size_tensor, padded_shapes=self.padded_feature_shape[mode])
        else:
            dataset = dataset.batch(self.batch_size_tensor)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _get_padded_shape(self, dataset, mode):
        for key in dataset:
            self.padded_feature_shape[mode][key] = dataset[key].shape
        return dataset

    def _combine_dataset(self, *dataset):
        combined_dict = {}
        for ds in dataset:
            for key in ds.keys():
                assert key not in combined_dict, "found duplicated key {} in unpaird feature sets".format(key)
                combined_dict[key] = ds[key]
        return combined_dict

    def _execute_mode_ops(self, dataset, mode):
        num_filters = len(self.mode_filter_ops[mode])
        ops = self.mode_forward_ops[mode][0]
        dataset = dataset.map(lambda dataset: self._preprocess_fn(dataset, ops), num_parallel_calls=self.num_core)
        if num_filters > 0:
            for idx in range(num_filters):
                dataset = dataset.filter(
                    lambda dataset: self.mode_filter_ops[mode][idx].forward(dataset, {"mode": mode}))
                ops = self.mode_forward_ops[mode][idx + 1]
                if len(ops) > 0:
                    dataset = dataset.map(lambda dataset: self._preprocess_fn(dataset, ops),
                                          num_parallel_calls=self.num_core)
        return dataset

    def _preprocess_fn(self, feature, ops):
        data = None
        for op in ops:
            if op.inputs:
                data = self._get_inputs_from_key(feature, op.inputs)
            data = op.forward(data, {})
            if op.outputs:
                feature = self._write_outputs_to_key(feature, data, op.outputs)
        return feature

    def _write_outputs_to_key(self, feature, outputs_data, outputs_key):
        if isinstance(outputs_key, str):
            feature[outputs_key] = outputs_data
        else:
            for key, data in zip(outputs_key, outputs_data):
                feature[key] = data
        return feature

    def _get_inputs_from_key(self, feature, inputs_key):
        if isinstance(inputs_key, list):
            data = [feature[key] for key in inputs_key]
        elif isinstance(inputs_key, tuple):
            data = tuple([feature[key] for key in inputs_key])
        else:
            data = feature[inputs_key]
        return data

    def _decode_records(self, dataset, mode, idx):
        decoded_data = {}
        all_data = tf.io.parse_single_example(dataset, features=self.all_features[mode][idx])
        for feature in self.feature_name[mode][idx]:
            data = all_data[feature]
            if "str" in str(data.dtype) and "str" not in self.feature_dtype[mode][idx][feature]:
                data = tf.io.decode_raw(data, convert_tf_dtype(self.feature_dtype[mode][idx][feature]))
                data = tf.reshape(data, self.record_feature_shape[mode][idx][feature])
            if "int" in str(data.dtype):
                data = tf.cast(data, tf.int32)
            elif "str" not in self.feature_dtype[mode][idx][feature]:
                data = tf.cast(data, tf.float32)
            decoded_data[feature] = data
        return decoded_data

    def show_results(self, inputs=None, mode="train", num_steps=1):
        """
        Shows batches of tensor data

        Args:
            inputs: Directory for saving TFRecords
            num_batches: Number of batches to show
            mode: Mode for training ("train", "eval" or "both")

        Returns:
            A dictionary containing the batches data
        """
        data = []
        self._prepare(inputs=inputs)
        assert self._prepare_mode(mode=mode), "could not find record in {} for mode {}".format(inputs, mode)
        dataset = self._input_stream(mode)
        for i, example in enumerate(dataset.take(num_steps)):
            data.append(example)
        return data

    def benchmark(self, inputs=None, mode="train", num_steps=500, log_interval=100):
        """
        benchmark the pipeline processing speed during training

        Args:
            inputs: Directory for saving TFRecords
            mode: Mode for training ("train", "eval")
        """
        self._prepare(inputs=inputs)
        assert self._prepare_mode(mode=mode), "could not find record in {} for mode {}".format(inputs, mode)
        self.num_example_per_epoch = np.min(self.num_examples[mode])
        epoch = tf.convert_to_tensor(0)
        step = tf.convert_to_tensor(0)
        dataset = self._input_stream(state={"mode": mode, "epoch": epoch, "step": step})
        start = time.time()
        for _ in dataset.take(num_steps):
            if step.numpy() % log_interval == 0 and step.numpy() > 0:
                duration = time.time() - start
                example_per_sec = log_interval * self.batch_size_tensor.numpy() / duration
                tf.print("FastEstimator: Step: %d, Epoch: %d, Batch Size %d, Example/sec %.2f" % (step.numpy(), epoch.numpy(), self.batch_size_tensor.numpy(), example_per_sec))
                start = time.time()
            step += 1
