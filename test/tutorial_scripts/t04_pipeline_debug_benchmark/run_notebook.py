import os

if __name__ == "__main__":
    source_dir = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "tutorial"))
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_notebook.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    tutorial_name = "t04_pipeline_debug_benchmark" # tutorial name without .ipynb extension
    nb_in_file = os.path.join(source_dir, tutorial_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", tutorial_name + "_out.ipynb"))

    run_info = "" # running parameter 
    result = os.system("papermill {} {} {} -k nightly-build 2>> {}".format(nb_in_file, nb_out_file, run_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(nb_in_file))