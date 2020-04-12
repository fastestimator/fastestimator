import os

from fastestimator.test.nightly_util import get_source_dir_path

if __name__ == "__main__":
    """The template for running apphub Jupyter notebook sample.
    Users only need to fill the SELF-FILLED SECTION.
    """
    # =================================  SELF-FILLED SECTION  =====================================
    # The name of the running example file. (without training "_torch.py" )
    example_name = "pggan"

    # The training arguments
    # 1. Usually we set the epochs:2, batch_size:2, max_steps_per_epoch:10
    # 2. The expression for the following setup is "--epochs 2 --batch_size 2 --max_steps_per_epoch 10"
    # 3. The syntax of this expression is different from run_notebook.py
    train_info = "--epochs 3 --target_size 8 --max_steps_per_epoch 20"

    # Do you want to run "fastestimator test"? (bool)
    need_test = False
    # ==============================================================================================

    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_torch_stderr.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    source_dir = get_source_dir_path(__file__)
    py_file = os.path.join(source_dir, example_name + "_torch.py")
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if need_test:
        result += os.system("fastestimator test {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
