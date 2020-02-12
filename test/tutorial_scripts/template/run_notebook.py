import os

if __name__ == "__main__":
     """The template for running tutorial Jupyter notebook sample. Users only need to fill the sections with comments.
    """
    source_dir = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "tutorial"))
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_notebook.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    tutorial_name = "" # tutorial name without .ipynb extension (it is should be this file's parant dir name)
    nb_in_file = os.path.join(source_dir, tutorial_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", tutorial_name + "_out.ipynb"))

    run_info = "" # Running parameter
    # Say if we want to set a=5 nad b=10, the expression for this is "-p a 5 -p b 10"
    # The arguement will re-declare the variable right after the jupyter notebook cell with "parameters" tag (there must
    # be one and only cell with "parameters" tag)

    result = os.system("papermill {} {} {} -k nightly-build 2>> {}".format(nb_in_file, nb_out_file, run_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(nb_in_file))
