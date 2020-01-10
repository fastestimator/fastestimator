import os

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))

    example_name = "pggan_nihchestxray_128"
    source_dir = os.path.join(apphub_path, "image_generation", "pggan_nihchestxray")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_python.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)
        
    py_file = os.path.join(source_dir, example_name + ".py")

    train_info = "" # the python script is not parametrized for testing (run very slow) 
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
