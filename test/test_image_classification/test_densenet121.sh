#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_image_classification=${path_apphub}'image_classification/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters
cnt=0

filepath=${path_image_classification}'densenet121_cifar10/'
filename='densenet121_cifar10.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    exit 1
fi

nb_filename='densenet121_cifar10.ipynb'
nb_param_filename='/densenet121_cifar10_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'densenet121_cifar10_param'

if ipython  ${path_temp}'/densenet121_cifar10_param.py'; then
    ((cnt=cnt+1))
else
    exit 1
fi
rm -rf ${path_temp}/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/densenet121_cifar10_param.py'
exit 0