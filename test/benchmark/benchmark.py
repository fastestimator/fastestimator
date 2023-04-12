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
# ==============================================================================
import importlib
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import psutil
import yaml

import fastestimator as fe
from fastestimator.trace.trace import Trace
from fastestimator.util.util import get_num_devices


class MemoryMeasure(Trace):
    def on_batch_end(self, data):
        if self.system.log_steps and self.system.global_step and (self.system.global_step % self.system.log_steps == 0
                                                                  or self.system.global_step == 1):
            used = psutil.virtual_memory().used / 1e9
            data.write_with_log("memory_used", used)


class ApphubModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self.load_module()

    def load_module(self):
        try:
            self.module = importlib.import_module(self.module_name)
        except:
            raise Exception("Module can't be imported.", self.module_name)

    def config_validation(self, batch_size_per_gpu, parameters):
        available_args = inspect.getfullargspec(self.module.get_estimator).args

        if 'batch_size' in available_args:
            batch_size = batch_size_per_gpu * get_num_devices()
            parameters['batch_size'] = batch_size
        elif 'batch_size_per_gpu' in available_args:
            parameters['batch_size_per_gpu'] = batch_size_per_gpu

        if 'data_dir' in parameters:
            if parameters['data_dir'] is None:
                del parameters['data_dir']

        for key in parameters:
            assert key in available_args, "The argument {}  is not in list of expected arguments {} of get_estimator of {} module.".format(key, available_args, self.module_name)

        return parameters

    def get_estimator(self, batch_size_per_gpu, apphub_parameters):
        apphub_parameters = self.config_validation(batch_size_per_gpu, apphub_parameters)
        return self.module.get_estimator(**apphub_parameters)


class FindApphubModule:
    def __init__(self) -> None:
        self.available_framework = ['tf', 'torch']
        self.folder_name = Path(fe.__file__).parent.parent.joinpath('apphub').as_posix()
        sys.path.append(self.folder_name)
        self.available_apphubs = self.get_available_apphubs()

    def is_valid_framework_ext(self, filename: str) -> bool:
        return self.validate_extension(filename)[0]

    def get_frame_work(self, filename: str) -> Union[str, None]:
        return self.validate_extension(filename)[1]

    def validate_extension(self, filename: str) -> Tuple[bool, Union[str, None]]:
        for frame_work in self.available_framework:
            if filename.endswith('_{}.py'.format(frame_work)):
                return True, frame_work
        return False, None

    def get_available_apphubs(self) -> Dict[Any, Any]:

        available_apphubs = list(
            set([
                Path(root).as_posix() for root,
                _,
                files in os.walk(self.folder_name) for file_name in files if self.is_valid_framework_ext(file_name)
            ]))

        valid_apphubs = [
            folder_path for folder_path in available_apphubs
            if len(Path(folder_path).relative_to(self.folder_name).as_posix().split('/')) == 2
        ]

        available_apphub_modules = {}

        for apphub in valid_apphubs:
            apphub_name = apphub.split('/')[-1]
            framework_modules = {}
            for filename in os.listdir(apphub):
                is_valid, frame_work = self.validate_extension(filename)
                if is_valid:
                    module_location = Path(apphub).relative_to(self.folder_name).joinpath(
                        Path(filename).stem).as_posix()
                    framework_modules[frame_work] = '.'.join(module_location.split('/'))
            available_apphub_modules[apphub_name] = framework_modules

        return available_apphub_modules

    def load_module(self, apphub_name, framework):
        assert apphub_name in self.available_apphubs, "Provided apphub is not available.Please select one of the following, {}".format(self.available_apphubs.keys())
        assert framework in self.available_apphubs[apphub_name], "Provided framework is not available. Please select one of the following frameworks {} available for {}.".format(self.available_apphubs[apphub_name].keys(), apphub_name)
        return ApphubModule(self.available_apphubs[apphub_name][framework])


def get_list_average(input_list: List, lower: float = 0.2, upper: float = 0.8):
    """Get the mean of the list between lower and upper limit.

    Args:
        input_list (str): location to save output of the apphub.
        lower (float): name of the apphub.
        upper (float): speed of the apphub.
    """
    input_list.sort()

    length_list = len(input_list)

    if length_list == 0:
        return 0

    lower_ind = int(lower * length_list)
    upper_ind = int(upper * length_list)

    if lower_ind == upper_ind:
        return input_list[lower_ind]
    else:
        return np.mean(input_list[lower_ind:upper_ind])


def write_output(output_file: Union[str, None], apphub_name: str, framework: str, speed: List, memory: List) -> None:
    """write the summary output.

    Args:
        output_file (str): location to save output of the apphub.
        apphub_name (str): name of the apphub.
        framework (str): framework of the apphub.
        speed (float): speed of the apphub.
        memory (float): memory of the apphub
    """
    speed = float(get_list_average(speed))
    memory = np.max(memory)
    no_of_gpus = get_num_devices()

    apphub_performance = {
        'speed': '{:.2f} steps/sec'.format(speed), 'cpu_ram': '{:.2f} GB'.format(memory), 'no_of_gpus': no_of_gpus
    }

    if output_file is None:
        output_folder = tempfile.mkdtemp()
        output_file = os.path.join(output_folder, 'performance.yml')

    print("Output is saved at:", output_file)

    output_performance = {}
    if os.path.exists(output_file):
        output_performance = yaml.safe_load(open(output_file, 'r'))
        output_performance = output_performance if output_performance else {}

    if apphub_name in output_performance:
        output_performance[apphub_name][framework] = apphub_performance
    else:
        output_performance[apphub_name] = {framework: apphub_performance}

    with open(output_file, 'w') as f:
        yaml.dump(output_performance, f)


def fastestimator_run(apphub_name: str,
                      framework: str,
                      batch_size_per_gpu: int,
                      output_file: Union[str, None] = None,
                      **kwargs):
    """Running the benchmarking of the appphub.

    Args:
        apphub_name (str): name of the apphub to run the benchmark on.
        framework (str): framework to run the apphub on.(tf/torch)
        batch_size_per_gpu(int): batch size per gpu.
        output_file (str): file to save the output summary
        kwargs (Dict[str, Any], optional): Any other additional arguments provided by user.
    """
    # find the apphub file
    apphub_module = FindApphubModule().load_module(apphub_name, framework)

    # get the apphub estimator
    estimator = apphub_module.get_estimator(batch_size_per_gpu=batch_size_per_gpu, apphub_parameters=kwargs)

    estimator.traces.append(MemoryMeasure())
    summary = estimator.fit(summary=apphub_name)
    speed = list(summary.history['train']['steps/sec'].values())
    memory = list(summary.history['train']['memory_used'].values())
    write_output(output_file, apphub_name, framework, speed, memory)