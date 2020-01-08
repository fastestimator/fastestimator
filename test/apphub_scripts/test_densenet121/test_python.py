import os

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))

    example_name = "densenet121_cifar10"
    source_dir = os.path.join(apphub_path, "image_classification", "densenet121_cifar10")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_python.txt"))
    if os.path.exist(stderr_file):
        os.remove(stderr_file)
        
    py_file = os.path.join(source_dir, example_name + ".py")

    train_info = "--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None"
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
