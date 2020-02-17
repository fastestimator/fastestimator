import os

if __name__ == "__main__":
    """The template for running apphub Jupyter notebook sample. Users only need to fill the sections with comments.
    """
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))
    example_name = ""  # The name of the running example. It should match the python file name.

    source_dir = os.path.join(apphub_path, "", "")  # The path to that example directory
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_python.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    py_file = os.path.join(source_dir, example_name + ".py")

    train_info = ""  # The training arguments
    # Usually we set the epochs:2, batch_size:2, steps_per_epoch:10, validation_steps:5
    # The expression for the following setup is "--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5"
    # The syntax of this expression is different from run_notebook.py

    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
