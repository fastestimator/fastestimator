import os

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))
    
    example_name = "maml"
    source_dir = os.path.join(apphub_path, "meta_learning", "MAML")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_notebook.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", example_name + "_out.ipynb"))

    train_info = "-p EPOCHS 1 -p BATCH_SIZE 2 -p STEPS_PER_EPOCH 10"
    result = os.system("papermill {} {} {} -k nightly-build 2>> {}".format(nb_in_file, nb_out_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(nb_in_file))