#!/bin/bash
path_apphub='../apphub/'
path_image_classification=${path_apphub}'image_classification/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters
FILES=$(find ${path_apphub} -type f -name '*.py')
cnt=0

# LeNet (Adversarial)
echo -en '\n'
echo 'Image Classification'
echo 'LeNet Adversarial'
echo -en '\n'

filepath=${path_image_classification}'lenet_cifar10_adversarial/'
filename='lenet_cifar10_adversarial.py'

#run python file
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed for CIFAR 10 Image classification (LeNet)'
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
    echo 'Testing failed for LeNet'
    exit 0
fi
rm -rf /tmp/tmp*
rm ${path_temp}${nb_param_filename}
rm ${path_temp}'/lenet_cifar10_adversarial_param.py'

# DenseNet
echo -en '\n'
echo 'DenseNet'
echo -en '\n'

filepath=${path_image_classification}'densenet121_cifar10/'
filename='densenet121_cifar10.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on DenseNet Image classification'
    exit 0
fi

nb_filename='densenet121_cifar10.ipynb'
nb_param_filename='/densenet121_cifar10_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'densenet121_cifar10_param'

if ipython  ${path_temp}'/densenet121_cifar10_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing on DenseNet notebook failed'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/densenet121_cifar10_param.py'

# LeNet Mixup
echo -en '\n'
echo 'LeNet CIFAR 10 Mixup'
echo -en '\n'
filepath=${path_image_classification}'lenet_cifar10_mixup/'
filename='lenet_cifar10_mixup.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet mixup'
    exit 0
fi

nb_filename='lenet_cifar10_mixup.ipynb'
nb_param_filename='/lenet_cifar10_mixup_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'lenet_cifar10_mixup_param'

if ipython  ${path_temp}'/lenet_cifar10_mixup_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet Mixup notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/lenet_cifar10_mixup_param.py'

# LeNet MNIST
echo -en '\n'
echo '4. LeNet MNIST'
echo -en '\n'

filepath=${path_image_classification}'lenet_mnist/'
filename='lenet_mnist.py'

if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet MNIST'
    exit 0
fi

nb_filename='lenet_mnist.ipynb'
nb_param_filename='/lenet_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'lenet_mnist_param'

if ipython  ${path_temp}'/lenet_mnist_param.py'; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet MNIST notebook'
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/lenet_mnist_param.py'
exit 1