#!/bin/bash
path_apphub='../apphub/'
path_image_generation=${path_apphub}'image_generation/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters
FILES=$(find ${path_apphub} -type f -name '*.py')
cnt=0

echo -en '\n'
echo 'Image Generation'
echo 'CVAE MNIST'
echo -en '\n'

filepath=${path_image_generation}'cvae_mnist/'
filename='cvae_mnist.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on CVAE MNIST'
    exit 0
fi

nb_filename='cvae_mnist.ipynb'
nb_param_filename='/cvae_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'cvae_mnist_param'

if ipython  ${path_temp}'/cvae_mnist_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on CVAE MNIST notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/cvae_mnist_param.py'

# CycleGAN Horse2Zebra
echo -en '\n'
echo 'Image Generation'
echo 'CycleGAN Horse2Zebra'
echo -en '\n'

filepath=${path_image_generation}'cyclegan_horse2zebra/'
filename='cyclegan_horse2zebra.py'

if fastestimator train ${filepath}${filename} --epochs 2 --steps_per_epoch 10; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on CycleGAN Horse2Zebra'
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
    echo 'Testing failed on CycleGAN notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/cyclegan_param.py'

# DCGAN MNIST
echo -en '\n'
echo 'Image Generation'
echo 'DcGAN MNIST'
echo -en '\n'

filepath=${path_image_generation}'dcgan_mnist/'
filename='dcgan_mnist.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on DcGAN MNIST'
    exit 0
fi

nb_filename='dcgan_mnist.ipynb'
nb_param_filename='/dcgan_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info} -p saved_model_path 'gen_epoch_0_step_10.h5'
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'dcgan_mnist_param'

if ipython  ${path_temp}'/dcgan_mnist_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on DCGAN notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm ${path_temp}${nb_param_filename}
rm ${path_temp}'/dcgan_mnist_param.py'
exit 1