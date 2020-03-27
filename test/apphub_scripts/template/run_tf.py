import os

if __name__ == "__main__":
    """The template for running apphub Jupyter notebook sample.
    Users only need to fill the SELF-FILLED SECTION.
    """
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))

    # =================================  SELF-FILLED SECTION  =====================================
    # The name of the running example file. (without training "_tf.py" )
    example_name = ""

    # The path to that example directory
    source_dir = os.path.join(apphub_path, "", "")

    # The training arguments
    # 1. Usually we set the epochs:2, batch_size:2, max_steps_per_epoch:10
    # 2. The expression for the following setup is "--epochs 2 --batch_size 2 --max_steps_per_epoch 10"
    # 3. The syntax of this expression is different from run_notebook.py
    train_info = ""
    # ==============================================================================================

    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_tf_stderr.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    py_file = os.path.join(source_dir, example_name + "_tf.py")
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))