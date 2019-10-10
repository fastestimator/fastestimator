#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../apphub/'
test_image_classification=${DIR}'/test_image_classification/'
test_image_generation=${DIR}'/test_image_generation/'
test_image_styletransfer=${DIR}'/test_image_styletransfer/'
test_image_segmentation=${DIR}'/test_image_segmentation/'
test_nlp=${DIR}'/test_nlp/'
test_tabular=${DIR}'/test_tabular/'
tmpdir=$(dirname $(mktemp -u))

# Image Classification examples test
bash ${test_image_classification}'test_densenet121.sh'
densenet_code=$?

bash ${test_image_classification}'test_lenet_adversarial.sh'
lenet_adv_code=$?

bash ${test_image_classification}'test_lenet_mixup.sh'
lenet_mixup_code=$?

bash ${test_image_classification}'test_lenet_mnist.sh'
lenet_mnist_code=$?

# Image Generation examples test
bash ${test_image_generation}'test_cvae_mnist.sh'
cvae_code=$?

bash ${test_image_generation}'test_cyclegan.sh'
cyclegan_code=$?

bash ${test_image_generation}'test_dcgan_mnist.sh'
dcgan_code=$?

# Image Segmentation examples test
bash ${test_image_segmentation}'test_unet_cub200.sh'
unet_cub_code=$?

bash ${test_image_segmentation}'test_unet_montgomery.sh'
unet_mont_code=$?

#Image StyleTransfer examples test
bash ${test_image_styletransfer}'test_fst_coco.sh'
fst_code=$?

# NLP examples test
bash ${test_nlp}'test_lstm_imdb.sh'
lstm_code=$?

#Tabular examples test
bash ${test_tabular}'test_dnn_housing.sh'
dnn_code=$?

#Tutorials examples test
tutorial_res=$(bash ${DIR}/test_tutorials.sh)

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

printf '%s\n' "$tutorial_res"