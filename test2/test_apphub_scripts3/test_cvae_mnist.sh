#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_temp=$(dirname $(mktemp -u))'/'

filepath=${path_apphub}'image_generation/cvae_mnist/' 
fname='cvae_mnist'
fname_para=$fname'_param'

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters

if fastestimator train ${filepath}${fname}'.py' ${train_info}; then
    ((cnt=cnt+1))
else
    exit 1
fi

#inject a parameter cell
papermill --prepare-only ${filepath}${fname}'.ipynb'  ${path_temp}${fname_para}'.ipynb' ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${fname_para}'.ipynb' --output ${fname_para}

if ipython  ${path_temp}${fname_para}'.py'; then
    ((cnt=cnt+1))
else
    exit 1
fi
rm -rf ${path_temp}/tmp*
rm  ${path_temp}${fname_para}'.ipynb'
rm  ${path_temp}${fname_para}'.py'

exit 0