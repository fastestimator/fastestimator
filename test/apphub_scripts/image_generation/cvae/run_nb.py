import os
import time


def get_apphub_path():
    apphub_path = None
    current_dir = os.path.abspath(os.path.join(__file__, ".."))
    while current_dir != "/":
        current_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if "apphub" in os.listdir(current_dir):
            apphub_path = os.path.abspath(os.path.join(current_dir, "apphub"))
            break

    if apphub_path is None:
        raise OSError("Could not find the apphub directory")

    return apphub_path


def get_relative_path():
    current_dir = os.path.abspath(os.path.join(__file__, ".."))
    split = current_dir.split("apphub_scripts/")
    if len(split) == 1:
        raise OSError("This file need to be put inside apphub_scripts directory")

    return split[-1]


def get_source_dir_path():
    apphub_path = get_apphub_path()
    relative_dir_path = get_relative_path()
    source_dir_path = os.path.join(apphub_path, relative_dir_path)

    return source_dir_path


if __name__ == "__main__":
    """The template for running apphub Jupyter notebook sample.
    Users only need to fill the sections
    """
    # ====================================  SELF-FILLED SECTION  ===================================
    # The name of the running example file. (without training ".ipynb")
    example_name = "cvae"

    # The training arguments
    # 1. Usually we set the epochs:2, batch_size:2, max_steps_per_epoch:10
    # 2. The expression for the above setup is "-p epochs 2 -p batch_size 2 -p max_steps_per_epoch 10"
    # 3. The arguement will re-declare the variable right after the jupyter notebook cell with "parameters" tag (there \
    # must be one and only cell with "parameters" tag)
    train_info = "-p epochs 2 -p batch_size 2 -p max_steps_per_epoch 10"
    # ==============================================================================================

    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_nb_stderr.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    source_dir = get_source_dir_path()
    nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", example_name + "_out.ipynb"))

    result = os.system("papermill {} {} {} -k nightly_build 2>> {}".format(nb_in_file,
                                                                           nb_out_file,
                                                                           train_info,
                                                                           stderr_file))
    time.sleep(10)  # time for GPU releasing memory for next test

    if result:
        raise ValueError("{} fail".format(nb_in_file))
