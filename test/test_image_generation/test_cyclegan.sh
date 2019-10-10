#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_image_generation=${path_apphub}'image_generation/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters

filepath=${path_image_generation}'cyclegan_horse2zebra/'
filename='cyclegan_horse2zebra.py'

if fastestimator train ${filepath}${filename} --epochs 2 --steps_per_epoch 10; then
    ((cnt=cnt+1))
else
    exit 0
fi

nb_filename='cyclegan.ipynb'
nb_param_filename='/cyclegan_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} -p epochs 2 -p steps_per_epoch 10
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'cyclegan_param'

if ipython  ${path_temp}'/cyclegan_param.py'; then
    ((cnt=cnt+1))
else
    exit 0
fi
rm -rf ${path_temp}/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/cyclegan_param.py'
exit 1