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
    """The apphub module.Apphub specific functionality.
    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = self.load_module()

    def load_module(self):
        """Loading the apphub module.

        Returns:
            module: loaded apphub module./exception on error while importing.

        """
        try:
            return importlib.import_module(self.module_name)
        except:
            raise Exception("Module can't be imported.", self.module_name)

    def config_validation(self, batch_size_per_gpu: int, parameters: Dict) -> Any:
        """Validation the apphub configuration.

        Args:
            batch_size_per_gpu (int): The number of items per gpu.
            parameters (Dict): Apphub specific parameters.

        Returns:
            parameters (Dict): Updated apphub specific parameters.
        """
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

    def get_estimator(self, batch_size_per_gpu, apphub_parameters) -> Any:
        """Get apphub estimator.

        Args:
            batch_size_per_gpu (int): The number of items per gpu.
            apphub_parameters (Dict): Apphub specific parameters.

        Returns:
            Any: Loaded apphub estimator.
        """
        apphub_parameters = self.config_validation(batch_size_per_gpu, apphub_parameters)
        return self.module.get_estimator(**apphub_parameters)


class FindApphubModule:
    """ Find all the apphub located available for installed fastestimator.
    """
    def __init__(self) -> None:
        self.available_framework = ['tf', 'torch']
        self.folder_name = Path(__file__).parent.parent.parent.joinpath('apphub').as_posix()
        sys.path.append(self.folder_name)
        self.available_apphubs = self.get_available_apphubs()

    def is_valid_framework_ext(self, filename: str) -> bool:
        """Check whether the framework is extension is available.

        Args:
            filename (str): The location of the file.(_tf.py/_torch.py)

        Returns:
            bool: True if the framework is available based on file suffix.
        """
        return self.validate_extension(filename)[0]

    def get_frame_work(self, filename: str) -> Union[str, None]:
        """Get the framework.

        Args:
            filename (str): The location of the apphub file.

        Returns:
            Union[str, None]: The framework name is its valid/None.
        """
        return self.validate_extension(filename)[1]

    def validate_extension(self, filename: str) -> Tuple[bool, Union[str, None]]:
        """Verify the framework of the provided file.

        Args:
            filename (str): The location of the apphub file.

        Returns:
            Tuple[bool, Union[str, None]]: True is the framework is available, framework of the apphub file.
        """
        for frame_work in self.available_framework:
            if filename.endswith('_{}.py'.format(frame_work)):
                return True, frame_work
        return False, None

    def get_available_apphubs(self) -> Dict[Any, Any]:
        """Get all the apphubs available in fastestimator.

        Returns:
            Dict[Any, Any]: The information about the available apphubs and frameworks.
        """

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

    def load_module(self, apphub_name, framework) -> ApphubModule:
        """Load the apphub module.

        Args:
            apphub_name (str): The apphub name we need to run.
            framework (str): The framework of the apphub.

        Returns:
            ApphubModule: Loaded apphub module.
        """
        assert apphub_name in self.available_apphubs, "Provided apphub is not available.Please select one of the following, {}".format(self.available_apphubs.keys())
        assert framework in self.available_apphubs[apphub_name], "Provided framework is not available. Please select one of the following frameworks {} available for {}.".format(self.available_apphubs[apphub_name].keys(), apphub_name)
        return ApphubModule(self.available_apphubs[apphub_name][framework])


def get_list_average(input_list: List, lower: float = 0.2, upper: float = 0.8) -> float:
    """Get the mean of the list between lower and upper bound.

    Args:
        input_list (str): List of numbers..
        lower (float): lower percentile.
        upper (float): upper percentile.

    Returns:
        float: Mean value of sub list.
    """
    input_list.sort()

    length_list = len(input_list)

    if length_list == 0:
        return 0.0

    lower_ind = int(lower * length_list)
    upper_ind = int(upper * length_list)

    if lower_ind == upper_ind:
        return float(input_list[lower_ind])
    else:
        return np.mean(input_list[lower_ind:upper_ind])


def format_output(speed: List, memory: List) -> Dict[str, Any]:
    """Format the apphub benchmark.

    Args:
        speed (List): List of logged steps/sec during training.
        memory (List): List of memory utilized during training.

    Returns:
        apphub_benchmark (Dict[str, Any]): A dictionary of apphub benchmark.
    """
    speed = float(get_list_average(speed))
    memory = np.max(memory)
    no_of_gpus = get_num_devices()

    apphub_benchmark = {
        'speed': '{:.2f} steps/sec'.format(speed), 'cpu_ram': '{:.2f} GB'.format(memory), 'no_of_gpus': no_of_gpus
    }
    return apphub_benchmark


def read_output_yml(output_file: str) -> Dict:
    """Read the output yml.

    Args:
        output_file (Union[str, None]): Location of the output yaml file.

    Returns:
        output_benchmark(Dict): Information about already performed apphub benchmarking.
    """
    assert Path(output_file).suffix in ['.yaml', '.yml'], "Please provide a yaml file"

    print("Output is saved at:", output_file)

    output_benchmark = {}
    if os.path.exists(output_file):
        output_benchmark = yaml.safe_load(open(output_file, 'r'))
        output_benchmark = output_benchmark if output_benchmark else {}

    return output_benchmark


def write_output(output_file: str, apphub_name: str, framework: str, speed: List, memory: List) -> None:
    """write the summary output.

    Args:
        output_file (str): location to save output of the apphub.
        apphub_name (str): name of the apphub.
        framework (str): framework of the apphub.
        speed (float): speed of the apphub.
        memory (float): memory of the apphub
    """
    apphub_benchmark = format_output(speed, memory)

    output_benchmark = read_output_yml(output_file)

    if apphub_name in output_benchmark:
        output_benchmark[apphub_name][framework] = apphub_benchmark
    else:
        output_benchmark[apphub_name] = {framework: apphub_benchmark}

    with open(output_file, 'w') as f:
        yaml.dump(output_benchmark, f)


def fastestimator_run(apphub_name: str,
                      framework: str,
                      batch_size_per_gpu: int,
                      output_file: str = os.path.join(tempfile.mkdtemp(), 'report.yml'),
                      **kwargs) -> None:
    """Running the benchmarking of the appphub.

    Args:
        apphub_name (str): name of the apphub to run the benchmark on.
        framework (str): framework to run the apphub on.(tf/torch)
        batch_size_per_gpu (int): number of images to use per gpu.
        output_file (str): file to save the output summary
        kwargs (Dict[str, Any], optional): Any other additional apphub specific arguments.
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
