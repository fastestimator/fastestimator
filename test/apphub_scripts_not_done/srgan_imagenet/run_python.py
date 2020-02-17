import os

# the concerning part
# 1. model dependency. It need to run srresnet_imagenet first
# 2. dataset dependency: 
#   it need to download imageNet first, and so far we don't have api to download that 
#   it need to download other dataset for inference showcase http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
# 3. take too long to run: it take ~3 hr for C5 instance 

if __name__ == "__main__":
    apphub_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "apphub"))

    example_name = "srgan_imagenet"
    source_dir = os.path.join(apphub_path, "image_generation", "srgan_imagenet")
    stderr_file = os.path.abspath(os.path.join(__file__, "..", "run_python.txt"))
    if os.path.exists(stderr_file):
        os.remove(stderr_file)
        
    py_file = os.path.join(source_dir, example_name + ".py")

    train_info = "--epochs 1 --batch_size 2 --steps_per_epoch 10 --validation_steps 5"
    result = os.system("fastestimator train {} {} 2>> {}".format(py_file, train_info, stderr_file))

    if result:
        raise ValueError("{} fail".format(py_file))
