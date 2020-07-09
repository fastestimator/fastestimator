#!/bin/bash
set -e

full_path=$(realpath $0)
dir_path=$(dirname $full_path)
example_name="fst"

# The training arguments
# 1. Usually we set the epochs:2, batch_size:2, max_train_steps_per_epoch:10
# 2. The expression for the following setup is "--epochs 2 --batch_size 8 --max_train_steps_per_epoch 10"
# 3. The syntax of this expression is different from run_notebook.py
style_img_path="${dir_path}/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
train_info="--epochs 2 --batch_size 4 --max_train_steps_per_epoch 10 --style_img_path ${style_img_path}"

# Do you want to run "fastestimator test"? (bool)
need_test=0
# ==============================================================================================

source_dir="${dir_path/'test/apphub_scripts'/'apphub'}"
stderr_file="${dir_path}/run_torch_stderr.txt"
py_file="${source_dir}/${example_name}_torch.py"

fastestimator train $py_file $train_info 2> $stderr_file

if [ $need_test -eq 1 ]; then
    fastestimator test $py_file $train_info 2>> $stderr_file
fi
