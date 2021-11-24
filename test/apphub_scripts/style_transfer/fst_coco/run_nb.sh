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

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
example_name="fst"

# The training arguments
# 1. Usually we set the epochs:2, batch_size:2, train_steps_per_epoch:2
# 2. The expression for the above setup is "-p epochs 2 -p batch_size 8 -p train_steps_per_epoch 2"
# 3. The arguement will re-declare the variable right after the jupyter notebook cell with "parameters" tag (there \
# must be one and only cell with "parameters" tag)
style_img_path="${dir_path}/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
test_img_path="${dir_path}/panda.jpeg"
train_info=(-p epochs 2 -p batch_size 4 -p train_steps_per_epoch 2 -p style_img_path "${style_img_path}"
            -p test_img_path "${test_img_path}")

# ==============================================================================================

source_dir="${dir_path/'test/apphub_scripts'/'apphub'}"
stderr_file="${dir_path}/run_nb_stderr.txt"
nb_out="${dir_path}/${example_name}_out.ipynb"
nb_in="${source_dir}/${example_name}.ipynb"

papermill "$nb_in" "$nb_out" "${train_info[@]}" "$@" -k nightly_build 2> "$stderr_file"
