#!/bin/bash
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
# ===============================================================================
set -e
export TF_CPP_MIN_LOG_LEVEL=2

example_name="bert"

# The training arguments
# 1. Usually we set the epochs:2, batch_size:2, max_train_steps_per_epoch:2
# 2. The expression for the following setup is "--epochs 2 --batch_size 8 --max_train_steps_per_epoch 2"
# 3. The syntax of this expression is different from run_notebook.py
train_info="--epochs 2 --batch_size 4 --max_train_steps_per_epoch 2 --max_eval_steps_per_epoch 2"

# Do you want to run "fastestimator test"? (bool)
need_test=0
# ==============================================================================================

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"

source_dir="${dir_path/'test/apphub_scripts'/'apphub'}"
stderr_file="${dir_path}/run_tf_stderr.txt"
py_file="${source_dir}/${example_name}_tf.py"

fastestimator train "$py_file" "$train_info" "$@" 2> "$stderr_file"

if [ "$need_test" -eq 1 ]; then
    fastestimator test "$py_file" "$train_info" "$@" 2>> "$stderr_file"
    echo 'run test'
fi
