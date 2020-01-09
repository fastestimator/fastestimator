import os

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))
    
    example_name = "fst_coco"
    source_dir = os.path.join(apphub_path, "style_transfer", "fst_coco")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_notebook.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", example_name + "_out.ipynb"))

    train_info = "-p steps_per_epoch 10 -p saved_model_path style_transfer_net_epoch_0_step_10.h5 -p img_path {}".format(os.path.join(source_dir, "panda.jpeg"))
    result = os.system("papermill {} {} {} -k nightly-build 2>> {}".format(nb_in_file, nb_out_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(nb_in_file))