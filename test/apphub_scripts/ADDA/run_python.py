import os
import shutil

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))

    example_name = "ADDA"
    source_dir = os.path.join(apphub_path, "domain_adaptation", "ADDA")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_python.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)
        
    py_file = os.path.join(source_dir, example_name + ".py")
    pretrained_fe_path = os.path.join(source_dir, "feature_extractor.h5")
    classifier_path = os.path.join(source_dir, "classifier.h5")

    train_info = "--epochs 2 --pretrained_fe_path {} --classifier_path {}".format(pretrained_fe_path, classifier_path)
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    shutil.rmtree(os.path.join(os.path.expanduser("~"), "fastestimator_data", "MNIST")) # remove the dataset since it will affect other testing
    
    if result:
        raise ValueError("{} fail".format(py_file))
