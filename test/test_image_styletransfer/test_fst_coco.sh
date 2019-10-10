#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_image_styletransfer=${path_apphub}'image_styletransfer/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters
FILES=$(find ${path_apphub} -type f -name '*.py')
cnt=0

#Fst COCO
filepath=${path_image_styletransfer}'fst_coco/'
filename='fst_coco.py'

if fastestimator train ${filepath}${filename} --steps_per_epoch 10; then
    ((cnt=cnt+1))
else
    exit 0
fi

nb_filename='fst_coco.ipynb'
nb_param_filename='/fst_coco_param.ipynb'
#inject a parameter cell

papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} -p steps_per_epoch 10 -p img_path ${filepath}'panda.jpeg' -p saved_model_path 'style_transfer_net_epoch_0_step_10.h5'
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'fst_coco_param'

if ipython  ${path_temp}'/fst_coco_param.py'; then
    ((cnt=cnt+1))
else
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/fst_coco_param.py'
exit 1