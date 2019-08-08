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
import tensorflow as tf
import numpy as np
import json
import os


def get_number_of_examples(file_path, show_warning=True, compression=None):
    """
    Returns number of examples in one TFRecord

    Args:
        file_path: Path of TFRecord file
        show_warning: Whether to display warning message
        compression: None, 'GZIP' or 'ZLIB'

    Returns:
        Number of examples in the TFRecord
    """
    _, ext = os.path.splitext(file_path)
    assert "tfrecord" in ext, "please make sure data is in tfrecord format"
    dataset = tf.data.TFRecordDataset(file_path, compression_type=compression)
    example_size = len(next(
        iter(dataset)).numpy()) + 16  #from multiple observations, tfrecord adds 16 byte to each example
    file_size = os.stat(file_path).st_size
    if file_size % example_size != 0 and show_warning:
        print("FastEstimator-Warning: Can't accurately calculate number of examples")
    return max(file_size // example_size, 1)


def get_features(file_path, compression=None):
    """
    Returns the feature information in TFRecords

    Args:
        file_path: Path of TFRecord file
        compression: None, 'GZIP' or 'ZLIB'

    Returns:
        Dictionary containing feature information of TFRecords
    """
    def _get_dtype(example, feature):
        dtype = list(example.features.feature[feature].DESCRIPTOR.fields_by_name.keys())
        dtype = np.array(dtype)
        feature_dtype = dtype[[example.features.feature[feature].HasField(x) for x in dtype]]
        feature_dtype = str(np.squeeze(feature_dtype))
        type_dict = {'bytes_list': tf.string, 'int64_list': tf.int64, 'float_list': tf.float32}
        tf_type = type_dict[feature_dtype]
        return tf_type

    _, ext = os.path.splitext(file_path)
    assert "tfrecord" in ext, "please make sure data is in tfrecord format"
    dataset = tf.data.TFRecordDataset(file_path, compression_type=compression)
    example = tf.train.Example.FromString(next(iter(dataset)).numpy())
    feature_list = list(example.features.feature.keys())
    tf_type_list = [tf.io.FixedLenFeature([], _get_dtype(example, f)) for f in feature_list]
    keys_to_features = dict(zip(feature_list, tf_type_list))
    return keys_to_features


def add_summary(data_dir, train_prefix, feature_name, feature_dtype, feature_shape, eval_prefix=None,
                num_train_examples=None, num_eval_examples=None, compression=None):
    """Adds summary.json file to existing path with tfrecords.

    Args:
        data_dir (str): Folder path where tfrecords are stored.
        train_prefix (str): The prefix of all training tfrecord files.
        feature_name (list): Feature name in the tfrecord in a list.
        feature_dtype (list): Original data type for specific feature, this is used for decoding purpose.
        feature_shape (list): Original data shape for specific feature, this is used for reshaping purpose.
        eval_prefix (str, optional): The prefix of all evaluation tfrecord files. Defaults to None.
        num_train_examples (int, optional): The total number of training examples, if None, it will calculate automatically. Defaults to None.
        num_eval_examples (int, optional): The total number of validation examples, if None, it will calculate automatically. Defaults to None.
        compression (str, optional): None, 'GZIP' or 'ZLIB'. Defaults to None.
    """
    train_files = [f for f in os.listdir(data_dir) if f.startswith(train_prefix)]
    assert len(train_files) > 0, "Couldn't find any training tfrecord files in %s" % data_dir
    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, train_files[0]), compression_type=compression)
    example = tf.train.Example.FromString(next(iter(dataset)).numpy())
    feature_list = list(example.features.feature.keys())
    assert set(feature_list).issuperset(
        set(feature_name)
    ), "feature name should at least be subset of feature name in tfrecords, found %s , given %s." % (str(feature_list),
                                                                                                      str(feature_name))
    if not num_train_examples:
        # num_train_examples = [get_number_of_examples(os.path.join(data_dir, f)) for f in train_files]
        num_trian_files = len(train_files)
        logging_interval = max(num_trian_files // 10, 1)
        num_train_examples = []
        for i in range(num_trian_files):
            if (i + 1) % logging_interval == 0:
                print("FastEstimator: Calculating number of examples for train %d/%d" % (i + 1, num_trian_files))
            num_train_examples.append(
                get_number_of_examples(file_path=os.path.join(data_dir, train_files[i]), show_warning=i == 0,
                                       compression=compression))
    summary = {
        "feature_name": feature_name,
        "feature_dtype": feature_dtype,
        "feature_shape": feature_shape,
        "train_files": train_files,
        "num_train_examples": num_train_examples
    }
    if eval_prefix:
        eval_files = [f for f in os.listdir(data_dir) if f.startswith(eval_prefix)]
        assert len(eval_files) > 0, "Couldn't find any validation tfrecord files in %s" % data_dir
        if not num_eval_examples:
            # num_eval_examples = [get_number_of_examples(os.path.join(data_dir, f)) for f in eval_files]
            num_eval_files = len(eval_files)
            logging_interval = max(num_eval_files // 10, 1)
            num_eval_examples = []
            for i in range(num_eval_files):
                if (i + 1) % logging_interval == 0:
                    print("FastEstimator: Calculating number of examples for eval %d/%d" % (i + 1, num_eval_files))
                num_eval_examples.append(
                    get_number_of_examples(file_path=os.path.join(data_dir, eval_files[i]), show_warning=i == 0,
                                           compression=compression))
        summary["eval_files"] = eval_files
        summary["num_eval_examples"] = num_eval_examples
    if compression:
        summary["compression"] = compression
    with open(os.path.join(data_dir, "summary0.json"), 'w') as fp:
        json.dump(summary, fp, indent=4)
    print("FastEstimator: Writing " + str(os.path.join(data_dir, "summary0.json")))
