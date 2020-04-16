import os


def get_apphub_path(working_file: str) -> str:
    """Find the path to the apphub folder which is a sibling of the `working_file`.

    Args:
        working_file: A file within the same FastEstimator repository as apphub examples.

    Returns:
        The root path to the apphub folder.

    Raises:
        OSError: If the `working_file` does not correspond to any of the apphub file paths.
    """
    apphub_path = None
    current_dir = os.path.abspath(os.path.join(working_file, ".."))
    while current_dir != "/":
        current_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if "apphub" in os.listdir(current_dir):
            apphub_path = os.path.abspath(os.path.join(current_dir, "apphub"))
            break
    if apphub_path is None:
        raise OSError("Could not find the apphub directory")
    return apphub_path


def get_relative_path(working_file: str) -> str:
    """Convert an absolute path into a relative path within the apphub_scripts folder.

    Args:
        working_file: The absolute path to a test file.

    Returns:
        The relative path to the test file within the apphub_scripts folder.

    Raises:
        OSError: If the `working_file` is not located within the apphub_scripts folder.
    """
    current_dir = os.path.abspath(os.path.join(working_file, ".."))
    split = current_dir.split("apphub_scripts/")
    if len(split) == 1:
        raise OSError("This file need to be put inside apphub_scripts directory")
    return split[-1]


def get_source_dir_path(working_file: str) -> str:
    """Get the absolute path to the folder containing the files to be tested by the `working_file`.

    Args:
        working_file: The absolute path to a test file.

    Returns:
        The absolute path to the corresponding apphub directory.
    """
    apphub_path = get_apphub_path(working_file)
    relative_dir_path = get_relative_path(working_file)
    source_dir_path = os.path.join(apphub_path, relative_dir_path)
    return source_dir_path
