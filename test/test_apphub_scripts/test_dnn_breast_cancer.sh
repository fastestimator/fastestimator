#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../../apphub/'
path_temp=$(dirname $(mktemp -u))'/'

# example specific section
filepath=${path_apphub}'/tabular/dnn_breast_cancer/'
fname='dnn_breast_cancer'
fname_para=$fname'_param'

path_stderr_dir='stderr/'${fname}'/'
mkdir $path_stderr_dir

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters

if ! fastestimator train ${filepath}${fname}'.py' ${train_info} 2>> ${path_stderr_dir}'run_py.txt'; then
    exit 1
fi

#inject a parameter cell
papermill --prepare-only ${filepath}${fname}'.ipynb'  ${path_temp}${fname_para}'.ipynb' ${nb_train_info} 2>> ${path_stderr_dir}'run_papermill.txt'
jupyter nbconvert --to script  ${path_temp}${fname_para}'.ipynb' --output ${fname_para} 2>> ${path_stderr_dir}'run_convert.txt'

if ! ipython ${path_temp}${fname_para}'.py' 2>> ${path_stderr_dir}'run_ipy.txt'; then
    exit 1
fi
rm -rf ${path_temp}/tmp*
rm  ${path_temp}${fname_para}'.ipynb'
rm  ${path_temp}${fname_para}'.py'

exit 0