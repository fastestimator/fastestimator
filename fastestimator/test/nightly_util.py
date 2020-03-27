import os


def get_apphub_path(working_file):
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


def get_relative_path(working_file):
    current_dir = os.path.abspath(os.path.join(working_file, ".."))
    split = current_dir.split("apphub_scripts/")
    if len(split) == 1:
        raise OSError("This file need to be put inside apphub_scripts directory")

    return split[-1]


def get_source_dir_path(working_file):
    apphub_path = get_apphub_path(working_file)
    relative_dir_path = get_relative_path(working_file)
    source_dir_path = os.path.join(apphub_path, relative_dir_path)

    return source_dir_path
