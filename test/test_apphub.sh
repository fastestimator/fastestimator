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
fail=0 
report_file="report.txt"
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

echo all the test results
# image classification
echo -en '\n\n'
if [ $densenet_code -eq 1 ] ; then
    echo 'Densenet121 test failed' >> $report_file
    fail=1
else
    echo 'Densenet121 test passed' >> $report_file
fi

if [ $lenet_adv_code -eq 1 ] ; then
    echo 'LeNet Adversarial test failed' >> $report_file
    fail=1
else
    echo 'LeNet Adversarial test passed' >> $report_file
fi

if [ $lenet_mixup_code -eq 1 ] ; then
    echo 'LeNet Mixup test failed' >> $report_file
    fail=1
else
    echo 'LeNet Mixup test passed' >> $report_file
fi

if [ $lenet_mnist_code -eq 1 ] ; then
    echo 'LeNet MNIST test failed' >> $report_file
    fail=1
else
    echo 'LeNet MNIST test passed' >> $report_file
fi

# image generation
if [ $cvae_code -eq 1 ] ; then
    echo 'CVAE test failed' >> $report_file
    fail=1
else
    echo 'CVAE test passed' >> $report_file
fi

if [ $cyclegan_code -eq 1 ] ; then
    echo 'CycleGAN test failed' >> $report_file
    fail=1
else
    echo 'CycleGAN test passed' >> $report_file
fi

if [ $dcgan_code -eq 1 ] ; then
    echo 'DCGAN test failed' >> $report_file
    fail=1
else
    echo 'DCGAN test passed' >> $report_file
fi

# image segmentation
if [ $unet_cub_code -eq 1 ] ; then
    echo 'UNET Cub200 test failed' >> $report_file
    fail=1
else
    echo 'UNET Cub200 test passed' >> $report_file
fi

if [ $unet_mont_code -eq 1 ] ; then
    echo 'UNET Montgomery test failed' >> $report_file
    fail=1
else
    echo 'UNET Montgomery test passed' >> $report_file
fi

#Image StyleTransfer
if [ $fst_code -eq 1 ] ; then
    echo 'FST COCO test failed' >> $report_file
    fail=1
else
    echo 'FST COCO test passed' >> $report_file
fi

Image NLP
if [ $lstm_code -eq 1 ] ; then
    echo 'LSTM IMDB test failed' >> $report_file
    fail=1
else
    echo 'LSTM IMDB test passed' >> $report_file
fi

Image Tabular
if [ $dnn_code -eq 1 ] ; then
    echo 'DNN Housing test failed' >> $report_file
    fail=1
else
    echo 'DNN Housing test passed' >> $report_file
fi


#Tutorials examples test
bash ${DIR}'/test_tutorials.sh'
tutorial_res_code=$?

if [ $tutorial_res_code -eq 1 ] ; then
    fail=1
fi

cat $report_file
rm $report_file

if [ $fail -eq 1 ] ; then
    exit 1
else
    exit 0
fi
