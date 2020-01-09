import os
import shutil

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))
    
    example_name = "ADDA"
    source_dir = os.path.join(apphub_path, "domain_adaptation", "ADDA")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_notebook.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)
        
    nb_in_file = os.path.join(source_dir, example_name + ".ipynb")
    nb_out_file = os.path.abspath(os.path.join(__file__, "..", example_name + "_out.ipynb"))
    pretrained_fe_path = os.path.join(source_dir, "feature_extractor.h5")
    classifier_path = os.path.join(source_dir, "classifier.h5")

    train_info = "-p epochs 2 -p batch_size 4 -p model_path {} -p classifier_path {} ".format(pretrained_fe_path, classifier_path)
    result = os.system("papermill {} {} {} 2>> {}".format(nb_in_file, nb_out_file, train_info, stderr_file))

    shutil.rmtree(os.path.join(os.path.expanduser("~"), "fastestimator_data", "MNIST") # remove the dataset since it will affect other testing

    if result:
        raise ValueError("{} fail".format(nb_in_file))