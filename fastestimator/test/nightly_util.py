# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import os


def get_uncle_path(uncle_dir: str, working_file: str) -> str:
    """Find the path to the uncle folder of `working_file`.

    Args:
        uncle_dir: A target uncle folder
        working_file: A file within the same FastEstimator repository as apphub examples.

    Returns:
        The root path to the apphub folder.

    Raises:
        OSError: If the `working_file` does not correspond to any of the uncle paths.
    """
    uncle_path = None
    current_dir = os.path.abspath(os.path.join(working_file, ".."))
    while current_dir != "/":
        current_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if uncle_dir in os.listdir(current_dir):
            uncle_path = os.path.abspath(os.path.join(current_dir, uncle_dir))
            break
    if uncle_path is None:
        raise OSError("Could not find the {} directory".format(uncle_dir))
    return uncle_path


def get_relative_path(parent_dir: str, working_file: str) -> str:
    """Convert an absolute path into a relative path within the parent folder.

    Args:
        parent_dir: A parent folder
        working_file: The absolute path to a test file.

    Returns:
        The relative path to the test file within the parent_dir folder.

    Raises:
        OSError: If the `working_file` is not located within the parent_dir folder.
    """
    current_dir = os.path.abspath(os.path.join(working_file, ".."))
    split = current_dir.split("{}/".format(parent_dir))
    if len(split) == 1:
        raise OSError("This file need to be put inside {} directory".format(parent_dir))
    return split[-1]


def get_apphub_source_dir_path(working_file: str) -> str:
    """Get the absolute path to the apphub folder containing the files to be tested by the `working_file`.

    Args:
        working_file: The absolute path to a test file.

    Returns:
        The absolute path to the corresponding apphub directory.
    """
    apphub_path = get_uncle_path("apphub", working_file)
    relative_dir_path = get_relative_path("apphub_scripts", working_file)
    source_dir_path = os.path.join(apphub_path, relative_dir_path)
    return source_dir_path
