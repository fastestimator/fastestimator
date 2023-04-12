#!/bin/bash

# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
fastestimator run benchmark --apphub_name pyramidnet --framework tf --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml
sleep 5s
fastestimator run benchmark --apphub_name pyramidnet --framework torch --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml
sleep 5s
fastestimator run benchmark --apphub_name cyclegan --framework tf --batch_size 2 --train_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name cyclegan --framework torch --batch_size 2 --train_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name retinanet --framework tf --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name retinanet --framework torch --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name yolov5 --framework tf --batch_size_per_gpu 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name yolov5 --framework torch --batch_size_per_gpu 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name hrnet --framework tf --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name hrnet --framework torch --batch_size 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name solov2 --framework tf --batch_size_per_gpu 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s
fastestimator run benchmark --apphub_name solov2 --framework torch --batch_size_per_gpu 2 --train_steps_per_epoch 100 --eval_steps_per_epoch 100 --epochs 1 --output_file report.yml --data_dir /raid/shared_data
sleep 5s