import os


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
    Users only need to fill the SELF-FILLED SECTION.
    """
    # =================================  SELF-FILLED SECTION  =====================================
    # The name of the running example file. (without training "_torch.py" )
    example_name = "cifar10_fast"

    # The training arguments
    # 1. Usually we set the epochs:2, batch_size:2, max_steps_per_epoch:10
    # 2. The expression for the following setup is "--epochs 2 --batch_size 2 --max_steps_per_epoch 10"
    # 3. The syntax of this expression is different from run_notebook.py
    train_info = "--epochs 2 --batch_size 2 --max_steps_per_epoch 10"
    # ==============================================================================================

    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_torch_stderr.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    source_dir = get_source_dir_path()
    py_file = os.path.join(source_dir, example_name + "_torch.py")
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
