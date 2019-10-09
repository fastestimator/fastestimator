#!/bin/bash
path_apphub='../apphub/'
test_image_classification='/test_image_classification/'
test_image_generation='test_image_generation/'
test_image_styletransfer='test_image_styletransfer/'
test_image_segmentation='test_image_segmentation/'
test_nlp='/test_nlp/'
test_tabular='/test_tabular/'
tmpdir=$(dirname $(mktemp -u))

# Image Classification examples test
./${test_image_classification}'test_densenet121.sh'
densenet_code=$?

./${test_image_classification}'test_lenet_adversarial.sh'
lenet_adv_code=$?

./${test_image_classification}'test_lenet_mixup.sh'
lenet_mixup_code=$?

./${test_image_classification}'test_lenet_mnist.sh'
lenet_mnist_code=$?

# Image Generation examples test
./${test_image_generation}'test_cvae_mnist.sh'
cvae_code=$?

./${test_image_generation}'test_cyclegan.sh'
cyclegan_code=$?

./${test_image_generation}'test_dcgan_mnist.sh'
dcgan_code=$?

# Image Segmentation examples test
./${test_image_segmentation}'test_unet_cub200.sh'
unet_cub_code=$?

./${test_image_segmentation}'test_unet_montgomery.sh'
unet_mont_code=$?

#Image StyleTransfer examples test
./${test_image_styletransfer}'test_fst_coco.sh'
fst_code=$?

# NLP examples test
./${test_nlp}'test_lstm_imdb.sh'
lstm_code=$?

#Tabular examples test
./${test_tabular}'test_dnn_housing.sh'
dnn_code=$?

#Tabular examples test
./test_tutorials.sh
tutorial_code=$?

#echo all the test results
echo -en '\n\n'
if [ $densenet_code -eq 0 ] ; then
    echo 'Densenet121 test failed'
else
    echo 'Densenet121 test passed'
fi

if [ $lenet_adv_code -eq 0 ] ; then
    echo 'LeNet Adversarial test failed'
else
    echo 'LeNet Adversarial test passed'
fi

if [ $lenet_mixup_code -eq 0 ] ; then
    echo 'LeNet Mixup test failed'
else
    echo 'LeNet Mixup test passed'
fi

if [ $lenet_mnist_code -eq 0 ] ; then
    echo 'LeNet MNIST test failed'
else
    echo 'LeNet MNIST test passed'
fi

if [ $cvae_code -eq 0 ] ; then
    echo 'CVAE test failed'
else
    echo 'CVAE test passed'
fi

if [ $cyclegan_code -eq 0 ] ; then
    echo 'CycleGAN test failed'
else
    echo 'CycleGAN test passed'
fi

if [ $dcgan_code -eq 0 ] ; then
    echo 'DCGAN test failed'
else
    echo 'DCGAN test passed'
fi

if [ $unet_cub_code -eq 0 ] ; then
    echo 'UNET Cub200 test failed'
else
    echo 'UNET Cub200 test passed'
fi

if [ $unet_mont_code -eq 0 ] ; then
    echo 'UNET Montgomery test failed'
else
    echo 'UNET Montgomery test passed'
fi

if [ $fst_code -eq 0 ] ; then
    echo 'FST COCO test failed'
else
    echo 'FST COCO test passed'
fi

if [ $lstm_code -eq 0 ] ; then
    echo 'LSTM IMDB test failed'
else
    echo 'LSTM IMDB test passed'
fi

if [ $dnn_code -eq 0 ] ; then
    echo 'DNN Housing test failed'
else
    echo 'DNN Housing test passed'
fi

if [ $tutorial_code -eq 0 ] ; then
    echo 'Tutorials test failed'
else
    echo 'Tutorials test passed'
fi