import os
import papermill as pm 

apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "apphub"))

example_name = "dcgan_mnist"
source_dir = os.path.join(apphub_path, "image_generation", "dcgan_mnist")
test_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
stderr_dir = os.path.join(test_dir, "stderr", example_name)
stderr_file = os.path.join(stderr_dir, "run_python.txt")
nb_out_dir = os.path.join(test_dir, "nb_out")

if not os.path.exists(stderr_dir):
    os.makedirs(stderr_dir)

if not os.path.exists(nb_out_dir):
    os.makedirs(nb_out_dir)

py_file = os.path.join(source_dir, example_name + ".py")
nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
nb_out_file = os.path.join(nb_out_dir, example_name + "_out.ipynb")

# execute python file
train_info = "--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None"
os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

# execute jupyter-noteboook
pm.execute_notebook(nb_in_file, nb_out_file, {
    "epochs": 2,
    "batch_size": 2,
    "steps_per_epoch": 10,
    "validation_steps": 5,
    "saved_model_path": "gen_epoch_0_step_10.h5"
})