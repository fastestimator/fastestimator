from fastestimator.util.util import convert_tf_dtype
import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import os

class TFRecorder:
    """
    Class for creating TFRecords from numpy data or csv file containing paths to data on disk

    Args:
        train_data: Training dataset in the form of dictionary containing numpy data, or csv file (with file
            paths or data)
        feature_name: List of strings representing the feature names in the data (headers in csv, keys in dictionary
            or features in TFRecords)
        transform_dataset: List of lists of numpy transformations to be performed sequentially  on the raw data
            before the TFRecords are made.
        validation_data:  Validation data in the form of dictionary containing numpy data, or csv file, or fraction
            of training data to be sequestered for validation during training.
        create_patch: Whether to create multiple records from single example.
        max_tfrecord_mb: Maximum space to be occupied by one TFRecord.
        compression: tfrecords compression type, one of None, 'GZIP' or 'ZLIB'.
    """
    def __init__(self, train_data, feature_name, transform_dataset=None, validation_data=None, create_patch=False, max_tfrecord_mb=300, compression=None):
        self.train_data = train_data
        self.feature_name = feature_name
        self.transform_dataset = transform_dataset
        self.validation_data = validation_data
        self.create_patch = create_patch
        self.max_tfrecord_mb = max_tfrecord_mb
        self.num_process = 1
        self.rank = 0
        self.local_rank = 0
        self.num_subprocess = mp.cpu_count()
        self.num_eval_example_list = []
        self.num_train_example_list = []
        self.compression = compression
        option_map = {"GZIP": tf.python_io.TFRecordCompressionType.GZIP,
                      "ZLIB": tf.python_io.TFRecordCompressionType.ZLIB,
                      None: None}
        self.compression_option = option_map[compression]

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def create_tfrecord(self, save_dir=None):
        self._prepare_savepath(save_dir)
        self._verify_input()
        self._prepare_training_dict()
        self._get_feature_info()
        if self.validation_data:
            self._prepare_validation_dict()
            self.num_eval_example_list = self._write_tfrecord_parallel(self.validation_data, self.num_eval_exmaple_csv, "eval")
        self.num_train_example_list = self._write_tfrecord_parallel(self.train_data, self.num_train_example_csv, "train")
        self._write_json_summary()
        return self.save_dir

    def _get_feature_info(self):
        feature = self._transform_one_slice(self.train_data, self.rank)
        self.feature_name_new = list(feature.keys())
        self.mb_per_csv_example = 0
        self.feature_type_new = []
        for name in self.feature_name_new:
            data = np.asarray(feature[name])
            self.mb_per_csv_example += data.nbytes / 1e6
            dtype = str(data.dtype)
            if "<U" in dtype:
                dtype = "string"
            self.feature_type_new.append(dtype)

    def _write_tfrecord_parallel(self, dictionary, num_example_csv, mode):
        num_example_list = []
        processes = []
        queue = mp.Queue()
        num_example_process = num_example_csv // self.num_process
        process_start = self.rank * num_example_process
        if self.rank == (self.num_process -1):
            process_end = num_example_csv
        else:
            process_end = process_start + num_example_process
        num_example_process = process_end - process_start
        num_files_subprocess = int(np.ceil(num_example_process * self.mb_per_csv_example / self.max_tfrecord_mb / self.num_subprocess))
        file_idx_start_process = self.rank * num_files_subprocess * self.num_subprocess
        num_example_subprocess_remain = num_example_process % self.num_subprocess
        serial_start = process_start
        for i in range(self.num_subprocess):
            file_idx_start_subprocess = i * num_files_subprocess + file_idx_start_process
            if i < num_example_subprocess_remain:
                num_example_subprocess = num_example_process // self.num_subprocess + 1
            else:
                num_example_subprocess = num_example_process // self.num_subprocess
            serial_end = serial_start + num_example_subprocess
            processes.append(mp.Process(target=self._write_tfrecord_serial, args=(dictionary, serial_start, serial_end, num_files_subprocess, file_idx_start_subprocess, mode, queue)))
            serial_start += num_example_subprocess
        for p in processes:
            p.start()
        for _ in range(self.num_subprocess):
            num_example_list.extend(queue.get(block=True))
        for p in processes:
            p.join()
        return num_example_list

    def _write_tfrecord_serial(self, dictionary, serial_start, serial_end, num_files_subprocess, file_idx_start, mode, queue):
        num_example_list = []
        num_csv_example_per_file = (serial_end - serial_start) // num_files_subprocess
        show_progress = serial_start == 0
        file_start = serial_start
        file_end = file_start + num_csv_example_per_file
        for i in range(num_files_subprocess):
            file_idx = i + file_idx_start
            filename = mode + str(file_idx) + ".tfrecord"
            if i == (num_files_subprocess - 1):
                file_end = serial_end
            num_example_list.append(self._write_single_file(dictionary, filename, file_start, file_end, serial_start, serial_end, show_progress, mode))
            file_start += num_csv_example_per_file
            file_end += num_csv_example_per_file
        queue.put(num_example_list)

    def _write_single_file(self, dictionary, filename, file_start, file_end, serial_start, serial_end, show_progress, mode):
        goal_number = serial_end - serial_start
        logging_interval = max(goal_number // 20, 1)
        num_example = 0
        with tf.python_io.TFRecordWriter(os.path.join(self.save_dir, filename), options= self.compression_option) as writer:
            for i in range(file_start, file_end):
                if (i - serial_start) % logging_interval == 0 and show_progress:
                    print("FastEstimator: --Converting %s TFRecords %f%%--" % (mode.capitalize(), (i - serial_start)/goal_number*100))
                feature = self._transform_one_slice(dictionary, i)
                if self.create_patch:
                    num_patches = self._verify_dict(feature)
                    for j in range(num_patches):
                        feature_patch = self._get_dict_slice(feature, j)
                        self._write_single_example(feature_patch, writer)
                        num_example += 1
                else:
                    self._write_single_example(feature, writer)
                    num_example += 1
        return filename, num_example

    def _transform_one_slice(self, dictionary, index):
        feature = self._get_dict_slice(dictionary, index)
        if self.transform_dataset:
            feature = self._preprocessing(feature)
        feature = self.edit_feature(feature)
        return feature

    def _write_single_example(self, dictionary, writer):
        feature_tfrecord = {}
        keys = dictionary.keys()
        for key in keys:
            data = dictionary[key]
            if type(data) is np.ndarray and data.size ==1:
                data = data.reshape(-1)[0]
            if type(data) is np.ndarray and data.size > 1:
                feature_tfrecord[key] = self._bytes_feature(data.tostring())
            elif "int" in str(type(data)):
                feature_tfrecord[key] = self._int64_feature(data)
            elif "float" in str(type(data)):
                feature_tfrecord[key] = self._float_feature(data)
            elif "str" in str(type(data)):
                feature_tfrecord[key] = self._bytes_feature(data.encode())
            else:
                raise ValueError("only supports either scalar, string, or numpy array, got %s" % str(type(data)))
        example = tf.train.Example(features=tf.train.Features(feature=feature_tfrecord))
        writer.write(example.SerializeToString())

    def _get_dict_slice(self, dictionary, index):
        feature_slice = dict()
        keys = dictionary.keys()
        for key in keys:
            feature_slice[key] = dictionary[key][index]
        return feature_slice

    def edit_feature(self, feature):
        return feature

    def _preprocessing(self, feature):
        preprocessed_data = {}
        for idx in range(len(self.feature_name)):
            transform_list = self.transform_dataset[idx]
            feature_name = self.feature_name[idx]
            preprocess_data = feature[feature_name]
            for preprocess_obj in transform_list:
                preprocess_data = preprocess_obj.transform(preprocess_data, feature)
            preprocessed_data[feature_name] = preprocess_data
        return preprocessed_data

    def _verify_dict(self, dictionary):
        num_example_list = []
        for key in dictionary.keys():
            feature_data = dictionary[key]
            if type(feature_data) is list:
                num_example_list.append(len(feature_data))
            elif type(feature_data) is np.ndarray:
                num_example_list.append(feature_data.shape[0])
            else:
                raise ValueError("the feature only supports list or numpy array")
        assert len(set(num_example_list)) == 1, "features should have the same number of examples"
        return set(num_example_list).pop()

    def _prepare_training_dict(self):
        if type(self.train_data) is str:
            df = pd.read_csv(self.train_data)
            self.train_data = df.to_dict('list')
        self.num_train_example_csv = self._verify_dict(self.train_data)

    def _prepare_validation_dict(self):
        if type(self.validation_data) is float:
            num_example_takeout = int(self.validation_data * self.num_train_example_csv)
            train_idx = range(self.num_train_example_csv)
            eval_idx = np.random.choice(train_idx, num_example_takeout, replace=False)
            train_idx = np.delete(train_idx, eval_idx)
            self.validation_data = {}
            for key in self.feature_name:
                total_data = self.train_data[key]
                if type(total_data) is list:
                    self.train_data[key] = [total_data[x] for x in train_idx]
                    self.validation_data[key] = [total_data[x] for x in eval_idx]
                else:
                    self.train_data[key] = total_data[train_idx]
                    self.validation_data[key] = total_data[eval_idx]
            self.num_train_example_csv = self.num_train_example_csv - num_example_takeout
        elif type(self.validation_data) is str:
            df = pd.read_csv(self.validation_data)
            self.validation_data = df.to_dict('list')
        self.num_eval_exmaple_csv = self._verify_dict(self.validation_data)

    def _prepare_savepath(self, save_dir):
        if not save_dir:
            save_dir = os.path.join(tempfile.gettempdir(), "FEdataset")
        self.save_dir = save_dir
        if self.local_rank == 0:
            print("FastEstimator: Saving tfrecord to %s" % self.save_dir)
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
        if self.num_process > 1:
            import horovod.tensorflow.keras as hvd
            hvd.allreduce([0], name="Barrier")

    def _verify_input(self):
        assert type(self.train_data) is dict or self.train_data.endswith(".csv"), "train data should be either dictionary or csv file"
        if self.validation_data:
            assert type(self.validation_data) in [dict, float] or self.validation_data.endswith(".csv"), "validation data supports partition ratio (float), csv file or dictionary"

    def _write_json_summary(self):
        summary = {"feature_name": self.feature_name_new, "feature_dtype": self.feature_type_new}
        train_files, num_train_examples = zip(*self.num_train_example_list)
        summary["train_files"] = list(train_files)
        summary["num_train_examples"] = list(num_train_examples)
        if self.compression:
            summary["compression"] = self.compression
        if len(self.num_eval_example_list) > 0:
            eval_files, num_eval_examples = zip(*self.num_eval_example_list)
            summary["eval_files"] = list(eval_files)
            summary["num_eval_examples"] = list(num_eval_examples)
        file_name = "summary%d.json" % self.rank
        with open(os.path.join(self.save_dir, file_name), 'w') as fp:
            json.dump(summary, fp, indent=4)

def tfrecord_to_np(file_path):
    """
    Converts 1 TFRecord (created using fastestimator) to numpy data

    Args:
        file_path: Path of TFRecord file

    Returns:
        Dictionary containing numpy data
    """
    tensor_data = dict()
    folder_path = os.path.dirname(file_path)
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
    assert len(json_files) > 0, "Cannot find summary json file, you can either add the json file or use TFRecorder to create tfrecord"
    summary = json.load(open(json_files[0], 'r'))
    keys_to_features = get_features(file_path)
    decode_type = {name:dtype for (name, dtype) in zip(summary["feature_name"], summary["feature_dtype"])}
    for tf_record_example in tf.python_io.tf_record_iterator(file_path):
        example = tf.parse_single_example(tf_record_example, features=keys_to_features)
        for key in summary["feature_name"]:
            data = example[key]
            if "string" in str(data.dtype):
                data = tf.decode_raw(data, convert_tf_dtype(decode_type[key]))
            if key in tensor_data:
                tensor_data[key].append(data)
            else:
                tensor_data[key] = [data]
    with tf.Session() as sess:
        np_data = sess.run(tensor_data)
    for key in np_data.keys():
        np_data[key] = np.array(np_data[key])
    return np_data

def get_number_of_examples(file_path, show_warning=True, compression=None):
    """
    Returns number of examples in 1 TFRecord

    Args:
        file_path: Path of TFRecord file
        show_warning: Whether to display warning message
        compression: None, 'GZIP' or 'ZLIB'

    Returns:
        Number of examples in the TFRecord
    """
    _, ext = os.path.splitext(file_path)
    assert "tfrecord" in ext, "please make sure data is in tfrecord format"
    option_map = {"GZIP": tf.python_io.TFRecordCompressionType.GZIP,
                  "ZLIB": tf.python_io.TFRecordCompressionType.ZLIB,
                  None: None}
    iterator = tf.python_io.tf_record_iterator(file_path, options=option_map[compression])
    example_size = len(next(iterator)) + 16  #from multiple observations, tfrecord adds 16 byte to each example
    file_size = os.stat(file_path).st_size
    if file_size % example_size != 0 and show_warning:
        print("FastEstimator-Warning: Can't accurately calculate number of examples")
    return max(file_size//example_size, 1)

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
    
    option_map = {"GZIP": tf.python_io.TFRecordCompressionType.GZIP,
                  "ZLIB": tf.python_io.TFRecordCompressionType.ZLIB,
                  None: None}
    _, ext = os.path.splitext(file_path)
    assert "tfrecord" in ext, "please make sure data is in tfrecord format"
    iterator = tf.python_io.tf_record_iterator(file_path, options=option_map[compression])
    example = tf.train.Example.FromString(next(iterator))
    feature_list = list(example.features.feature.keys())
    tf_type_list = [tf.FixedLenFeature([], _get_dtype(example, f)) for f in feature_list]
    keys_to_features = dict(zip(feature_list, tf_type_list))
    return keys_to_features

def add_summary(data_dir, train_prefix, feature_name, feature_dtype, eval_prefix=None, num_train_examples=None, num_eval_examples=None, compression=None):
    """Adds summary.json file to existing path with tfrecords.

    Args:
        data_dir (str): Folder path where tfrecords are stored.
        train_prefix (str): The prefix of all training tfrecord files.
        feature_name (list): Feature name in the tfrecord in a list.
        feature_dtype (list): Original data type for specific feature, this is used for decoding purpose.
        eval_prefix (str, optional): The prefix of all evaluation tfrecord files. Defaults to None.
        num_train_examples (int, optional): The total number of training examples, if None, it will calculate automatically. Defaults to None.
        num_eval_examples (int, optional): The total number of validation examples, if None, it will calculate automatically. Defaults to None.
        compression (str, optional): None, 'GZIP' or 'ZLIB'. Defaults to None.
    """
    train_files = [f for f in os.listdir(data_dir) if f.startswith(train_prefix)]
    assert len(train_files) > 0, "Couldn't find any training tfrecord files in %s" % data_dir
    option_map = {"GZIP": tf.python_io.TFRecordCompressionType.GZIP,
                  "ZLIB": tf.python_io.TFRecordCompressionType.ZLIB,
                  None: None}
    iterator = tf.python_io.tf_record_iterator(os.path.join(data_dir, train_files[0]), options=option_map[compression])
    example = tf.train.Example.FromString(next(iterator))
    feature_list = list(example.features.feature.keys())
    assert set(feature_list).issuperset(set(feature_name)), "feature name should at least be subset of feature name in tfrecords, found %s , given %s." % (str(feature_list), str(feature_name))
    if not num_train_examples:
        # num_train_examples = [get_number_of_examples(os.path.join(data_dir, f)) for f in train_files]
        num_trian_files = len(train_files)
        logging_interval = max(num_trian_files//10, 1)
        num_train_examples = []
        for i in range(num_trian_files):
            if (i+1) % logging_interval == 0:
                print("FastEstimator: Calculating number of examples for train %d/%d" % (i+1, num_trian_files))
            num_train_examples.append(get_number_of_examples(file_path=os.path.join(data_dir, train_files[i]), show_warning=i==0, compression=compression))
    summary = {"feature_name": feature_name, "feature_dtype": feature_dtype, "train_files":train_files, "num_train_examples": num_train_examples}
    if eval_prefix:
        eval_files = [f for f in os.listdir(data_dir) if f.startswith(eval_prefix)]
        assert len(eval_files) > 0, "Couldn't find any validation tfrecord files in %s" % data_dir
        if not num_eval_examples:
            # num_eval_examples = [get_number_of_examples(os.path.join(data_dir, f)) for f in eval_files]
            num_eval_files = len(eval_files)
            logging_interval = max(num_eval_files//10, 1)
            num_eval_examples = []
            for i in range(num_eval_files):
                if (i+1) % logging_interval == 0:
                    print("FastEstimator: Calculating number of examples for eval %d/%d" % (i+1, num_eval_files))
                num_eval_examples.append(get_number_of_examples(file_path=os.path.join(data_dir, eval_files[i]), show_warning=i==0, compression=compression))
        summary["eval_files"] = eval_files
        summary["num_eval_examples"] = num_eval_examples
    if compression:
        summary["compression"] = compression
    with open(os.path.join(data_dir, "summary0.json"), 'w') as fp:
        json.dump(summary, fp, indent=4)
    print("FastEstimator: Writing " + str(os.path.join(data_dir, "summary0.json")))