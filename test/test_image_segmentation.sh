#!/bin/bash
path_apphub='../apphub/'
path_image_segmentation=${path_apphub}'image_segmentation/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters
FILES=$(find ${path_apphub} -type f -name '*.py')
cnt=0

# UNET Cub200
echo -en '\n'
echo 'Image Segmentation'
echo 'Unet Cub 200'
echo -en '\n'

filepath=${path_image_segmentation}'unet_cub200/'
filename='unet_cub200.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on Unet cub 200'
    exit 0
fi

nb_filename='unet_cub200.ipynb'
nb_param_filename='/unet_cub200_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'unet_cub200_param'

if ipython  ${path_temp}'/unet_cub200_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on UNET Cub200 notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/unet_cub200_param.py'

# UNET Montgomery
echo -en '\n'
echo 'Image Segmentation'
echo 'Unet Montgomery'
echo -en '\n'

filepath=${path_image_segmentation}'unet_montgomery/'
filename='unet_montgomery.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on Unet mongomery'
    exit 0
fi

nb_filename='unet_montgomery.ipynb'
nb_param_filename='/unet_montgomery_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'unet_montgomery_param'

if ipython  ${path_temp}'/unet_montgomery_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on UNET Montgomery notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/unet_montgomery_param.py'
exit 1