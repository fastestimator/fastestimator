import os
import papermill as pm 

apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "apphub"))

example_name = "ADDA"
source_dir = os.path.join(apphub_path, "domain_adaptation", "ADDA")
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
pretrained_fe_path = os.path.join(source_dir, "feature_extractor.h5")
classifier_path = os.path.join(source_dir, "classifier.h5")

# execute python file
train_info = "--epochs 2 --pretrained_fe_path {} --classifier_path {}".format(pretrained_fe_path, classifier_path)
os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

# execute jupyter-noteboook
nb_train_info = "-p epochs 2 -p batch_size 4"
pm.execute_notebook(nb_in_file, nb_out_file, {
    "epochs": 2,
    "batch_size": 4,
    "model_path": pretrained_fe_path,
    "classifier_path": classifier_path
})