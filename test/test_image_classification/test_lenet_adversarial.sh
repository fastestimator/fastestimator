#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_image_classification=${path_apphub}'image_classification/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters

filepath=${path_image_classification}'lenet_cifar10_adversarial/'
filename='lenet_cifar10_adversarial.py'

#run python file
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    exit 0
fi

nb_filename='lenet_cifar10_adversarial.ipynb'
nb_param_filename='/lenet_cifar10_adversarial_param.ipynb'
#inject a parameter cell and convert to python script
papermill --prepare-only ${filepath}${nb_filename} ${path_temp}${nb_param_filename} ${nb_train_info} -p num_test_samples 10
jupyter nbconvert --to script ${path_temp}${nb_param_filename} --output 'lenet_cifar10_adversarial_param'

if ipython ${path_temp}'/lenet_cifar10_adversarial_param.py'; then
    ((cnt=cnt+1))
else
    exit 0
fi
rm -rf ${path_temp}/tmp*
rm ${path_temp}${nb_param_filename}
rm ${path_temp}'/lenet_cifar10_adversarial_param.py'
exit 1