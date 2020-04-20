import os
import time

from fastestimator.test.nightly_util import get_source_dir_path

if __name__ == "__main__":
    """The template for running apphub Jupyter notebook sample.
    Users only need to fill the sections
    """
    # ====================================  SELF-FILLED SECTION  ===================================
    # The name of the running example file. (without training ".ipynb")
    example_name = "fst"

    # The training arguments
    # 1. Usually we set the epochs:2, batch_size:2, max_steps_per_epoch:10
    # 2. The expression for the above setup is "-p epochs 2 -p batch_size 2 -p max_steps_per_epoch 10"
    # 3. The arguement will re-declare the variable right after the jupyter notebook cell with "parameters" tag (there \
    # must be one and only cell with "parameters" tag)
    style_img_path = os.path.abspath(os.path.join(__file__, "..", "Vassily_Kandinsky,_1913_-_Composition_7.jpg"))
    test_img_path = os.path.abspath(os.path.join(__file__, "..", "panda.jpeg"))
    train_info = "-p epochs 2 -p batch_size 2 -p max_steps_per_epoch 10 -p style_img_path {} -p test_img_path {}".format(
        style_img_path, test_img_path)
    # ==============================================================================================

    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_nb_stderr.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    source_dir = get_source_dir_path(__file__)
    nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", example_name + "_out.ipynb"))

    result = os.system("papermill {} {} {} -k nightly_build 2>> {}".format(nb_in_file,
                                                                           nb_out_file,
                                                                           train_info,
                                                                           stderr_file))
    time.sleep(10)  # time for GPU releasing memory for next test

    if result:
        raise ValueError("{} fail".format(nb_in_file))
