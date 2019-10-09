#!/bin/bash
path_apphub='../apphub/'
tmpdir=$(dirname $(mktemp -u))

# Image Classification examples test
./test_image_classification.sh
img_class_code=$?

# Image Generation examples test
./test_image_generation.sh
img_gen_code=$?

# Image Segmentation examples test
./test_image_segmentation.sh
img_seg_code=$?

#Image StyleTransfer examples test
./test_image_styletransfer.sh
img_style_code=$?

# NLP examples test
./test_nlp.sh
nlp_retn_code=$?

#Tabular examples test
./test_tabular.sh
tabular_code=$?

#Tabular examples test
./test_tutorials.sh
tutorial_code=$?

#echo all the test results
echo -en '\n\n'
if [ $img_class_code -eq 0 ] ; then
    echo 'Image Classification examples tests failed'
else
    echo 'Image Classification examples tests passed'
fi

if [ $img_seg_code -eq 0 ] ; then
    echo 'Image Segmentation examples tests failed'
else
    echo 'Image Segmentation examples tests passed'
fi

if [ $img_gen_code -eq 0 ] ; then
    echo 'Image Generation examples tests failed'
else
    echo 'Image Generation examples tests passed'
fi

if [ $img_style_code -eq 0 ] ; then
    echo 'Image Styletransfer examples tests failed'
else
    echo 'Image Styletransfer examples tests passed'
fi

if [ $nlp_retn_code -eq 0 ] ; then
    echo 'NLP examples tests failed'
else
    echo 'NLP examples tests passed'
fi

if [ $tabular_code -eq 0 ] ; then
    echo 'Tabular examples tests failed'
else
    echo 'Tabular examples tests passed'
fi

if [ $tutorial_code -eq 0 ] ; then
    echo 'Tutorial tests failed'
else
    echo 'Tutorial examples tests passed'
fi