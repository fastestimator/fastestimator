#!/bin/bash
path_apphub='../apphub/'

train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10'
FILES=$(find ${path_apphub} -type f -name '*.py')
cnt = 0

for filename in $FILES; do
    echo $(basename $filename)
    echo $filename
done

echo -en '\n'
echo 'Image Classification'
echo '1. LeNet Adversarial'
echo -en '\n'
image_classification_path=${path_apphub}'image_classification/'
filepath=${image_classification_path}'lenet_cifar10_adversarial/'
filename='lenet_cifar10_adversarial.py'
echo ${filepath}${filename}
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed for CIFAR 10 Image classification (LeNet)'
fi
nb_filename='lenet_cifar10_adversarial.ipynb'
nb_param_filename='lenet_cifar10_adversarial_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info} -p num_test_samples 10
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'lenet_cifar10_adversarial_param'

if ipython ${filepath}'lenet_cifar10_adversarial_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'lenet_cifar10_adversarial_param.py'

echo -en '\n'
echo '2. DenseNet'
echo -en '\n'
filepath=${image_classification_path}'densenet121_cifar10/'
filename='densenet121_cifar10.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on DenseNet Image classification'
fi

nb_filename='densenet121_cifar10.ipynb'
nb_param_filename='densenet121_cifar10_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'densenet121_cifar10_param'

if ipython ${filepath}'densenet121_cifar10_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'densenet121_cifar10_param.py'

echo -en '\n'
echo '3. LeNet CIFAR 10 Mixup'
echo -en '\n'
filepath=${image_classification_path}'lenet_cifar10_mixup/'
filename='lenet_cifar10_mixup.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet mixup'
fi
nb_filename='lenet_cifar10_mixup.ipynb'
nb_param_filename='lenet_cifar10_mixup_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'lenet_cifar10_mixup_param'

if ipython ${filepath}'lenet_cifar10_mixup_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'lenet_cifar10_mixup_param.py'

echo -en '\n'
echo '4. LeNet MNIST'
echo -en '\n'
filepath=${image_classification_path}'lenet_mnist/'
filename='lenet_mnist.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on LeNet MNIST'
fi
nb_filename='lenet_mnist.ipynb'
nb_param_filename='lenet_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'lenet_mnist_param'

if ipython ${filepath}'lenet_mnist_param.py'; then
    echo 'notebook pa'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'lenet_mnist_param.py'

echo -en '\n'
echo 'Image Detection'
echo '5. RetinaNet svhn'
echo -en '\n'
image_detection_path=${path_apphub}'image_detection/'
filepath=${image_detection_path}'retinanet_svhn/'
filename='retinanet_svhn.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on RetinaNet Svhn'
fi
nb_filename='retinanet_svhn.ipynb'
nb_param_filename='retinanet_svhn_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'retinanet_svhn_param'

if ipython ${filepath}'retinanet_svhn_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'retinanet_svhn_param.py'

echo -en '\n'
echo 'Image Generation'
echo '6. CVAE MNIST'
echo -en '\n'
image_generation_path=${path_apphub}'image_generation/'
filepath=${image_generation_path}'cvae_mnist/'
filename='cvae_mnist.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on CVAE MNIST'
fi
nb_filename='cvae_mnist.ipynb'
nb_param_filename='cvae_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'cvae_mnist_param'

if ipython ${filepath}'cvae_mnist_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'cvae_mnist_param.py'

echo -en '\n'
echo 'Image Generation'
echo '7. CycleGAN Horse2Zebra'
echo -en '\n'
filepath=${image_generation_path}'cyclegan_horse2zebra/'
filename='cyclegan_horse2zebra.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on CycleGAN Horse2Zebra'
fi
nb_filename='cyclegan.ipynb'
nb_param_filename='cyclegan_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'cyclegan_param'

if ipython ${filepath}'cyclegan_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'cyclegan_param.py'

echo -en '\n'
echo 'Image Generation'
echo '8. DcGAN MNIST'
echo -en '\n'
filepath=${image_generation_path}'dcgan_mnist/'
filename='dcgan_mnist.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on DcGAN MNIST'
fi
nb_filename='dcgan_mnist.ipynb'
nb_param_filename='dcgan_mnist_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'dcgan_mnist_param'

if ipython ${filepath}'dcgan_mnist_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'dcgan_mnist_param.py'

echo -en '\n'
echo 'Image Segmentation'
echo '9. Unet cub 200'
echo -en '\n'
image_segmentation_path=${path_apphub}'image_segmentation/'
filepath=${image_segmentation_path}'unet_cub200/'
filename='unet_cub200.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on Unet cub 200'
fi
nb_filename='unet_cub200.ipynb'
nb_param_filename='unet_cub200_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'unet_cub200_param'

if ipython ${filepath}'unet_cub200_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'unet_cub200_param.py'

echo -en '\n'
echo 'Image Segmentation'
echo '9. Unet montgomery'
echo -en '\n'
filepath=${image_segmentation_path}'unet_montgomery/'
filename='unet_montgomery.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on Unet mongomery'
fi
nb_filename='unet_montgomery.ipynb'
nb_param_filename='unet_montgomery_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'unet_montgomery_param'

if ipython ${filepath}'unet_montgomery_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'unet_montgomery_param.py'

echo -en '\n'
echo 'Image Styletransfer'
echo '10. Fst Cco'
echo -en '\n'
image_styletransfer_path=${path_apphub}'image_styletransfer/'
filepath=${image_styletransfer_path}'fst_coco/'
filename='fst_coco.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on image styletransfer'
fi
nb_filename='fst_coco.ipynb'
nb_param_filename='fst_coco_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'fst_coco_param'

if ipython ${filepath}'fst_coco_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'fst_coco_param.py'

echo -en '\n'
echo 'NLP'
echo '11. LSTM IMDB'
echo -en '\n'
nlp_path=${path_apphub}'NLP/'
filepath=${nlp_path}'lstm_imdb/'
filename='lstm_imdb.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on lstm imdb'
fi
nb_filename='lstm_imdb.ipynb'
nb_param_filename='lstm_imdb_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'lstm_imdb_param'

if ipython ${filepath}'lstm_imdb_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'lstm_imdb_param.py'

echo -en '\n'
echo 'Tabular'
echo '11. DNN housing'
echo -en '\n'
tabular_path=${path_apphub}'tabular/'
filepath=${tabular_path}'dnn_housing/'
filename='dnn_housing.py'
if fastestimator train ${filepath}${filename} ${train_info}; then
    ((cnt=cnt+1))
else
    echo 'Testing failed on dnn housing'
fi
nb_filename='dnn_housing.ipynb'
nb_param_filename='dnn_housing_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename} ${filepath}${nb_param_filename} ${nb_train_info}
jupyter nbconvert --to script ${filepath}${nb_param_filename} --output 'dnn_housing_param'

if ipython ${filepath}'dnn_housing_param.py'; then
    echo 'notebook passed'
else
    echo 'notebook failed'
fi
rm -rf /tmp/tmp*
rm ${filepath}${nb_param_filename}
rm ${filepath}'dnn_housing_param.py'
echo $cnt