# FastEstimator Application Hub

Welcome to FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art accessible to everyone.

## Purpose of Application Hub

* Provide place to learn implementation details of state-of-the-art
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own project/product


## Why not just learn from official implementation
If you ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000+ lines of code or 500+ lines of code to implement the same idea using different AI frameworks. In FastEstimator, we strive to make things simpler and more intuitive while preserving the flexibility. As a result, many state-of-the-art AI implementations can be simplified such that the code directly reflects the ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) include 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/tree/master/apphub/image_generation/pggan_nihchestxray) only uses 500+ lines.

To summarize, We spent the extra time learning from official implementation so you can save time by learning from us!

## What's included in each example

Each example in contains two files:

1. python file (.py): The FastEstimator source code needed to run the example.
2. jupyter notebook (.ipynb): notebook that provides step-by-step instructions and explanations about the implementation.


## How do I run each example

One can simply execute the python file of any example:
```
$ python lenet_mnist.py
```

or use our Command-Line Interface(CLI):

```
$ fastestimator train lenet_mnist.py
```

The benefit of CLI is allowing users to configure input args of `get_estimator`:

```
$ fastestimator train lenet_mnist.py --batch_size 64 --epochs 4
```


## Table of Contents:
* Natural Language Processing (NLP)

    * LSTM: Sentimental classification based on IMDB movie review. [[paper](https://www.bioinf.jku.at/publications/older/2604.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/lstm_imdb/lstm_imdb.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/lstm_imdb/lstm_imdb.ipynb)]

* Domain Adaptation
    * Adversarial Discriminative Domain Adaptation (ADDA): Adapting trained MNIST clasifier to USPS digits dataset without label data. [[paper](https://arxiv.org/abs/1702.05464)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/domain_adaptation/ADDA/ADDA.py)]

* Image Classification
    * DenseNet: Image classifier with DenseNet121 on Cifar10 dataset. [[paper](https://arxiv.org/abs/1608.06993)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/densenet121_cifar10/densenet121_cifar10.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/densenet121_cifar10/densenet121_cifar10.ipynb)]

    * Adversarial Attack: Image classifier training to prevent resist adversarial attaks on Cifar10 dataset. [[paper](https://arxiv.org/abs/1412.6572)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_adversarial/lenet_cifar10_adversarial.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_adversarial/lenet_cifar10_adversarial.ipynb)]

    * Mixup: Image classifier training that uses mix up as data augmentation strategy to enhance robustness on Cifar10 dataset. [[paper](https://arxiv.org/abs/1710.09412)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_mixup/lenet_cifar10_mixup.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_mixup/lenet_cifar10_mixup.ipynb)]

    * LeNet: Convolutional Neural Networks trained on Mnist dataset. [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_mnist/lenet_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_mnist/lenet_mnist.ipynb)]

* Image Generation
    * CVAE: Image feature representation learning and image generation with Convolutional Variational AutoEncoder on Mnist dataset. [[paper](https://arxiv.org/abs/1312.6114)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae_mnist/cvae_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae_mnist/cvae_mnist.ipynb)]

    * Cycle-GAN: Unpaired image-to-image translation between two domain using Cycle-GAN.[[paper](https://arxiv.org/abs/1703.10593)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan_horse2zebra/cyclegan_horse2zebra.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan_horse2zebra/cyclegan.ipynb)]

    * DCGAN: Image generation with Deep Convolutional GAN on Mnist dataset. [[paper](https://arxiv.org/abs/1511.06434)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan_mnist/dcgan_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan_mnist/dcgan_mnist.ipynb)]

    * PGGAN: High-resolution image generation with progressive growing of GANs on NIH chest X-ray dataset. [[paper](https://arxiv.org/abs/1710.10196)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan_nihchestxray/pggan_nihchestxray.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan_nihchestxray/pggan_nihchestxray.ipynb)]

* Image Segmentation
    * UNet: Lung segmentation with UNet using chest X-ray dataset. [[paper](https://arxiv.org/abs/1505.04597)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_segmentation/unet_montgomery/unet_montgomery.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_segmentation/unet_montgomery/unet_montgomery.ipynb)]

    * UNet: Image segmentation with UNET using CUB200 bird dataset. [[paper](https://arxiv.org/abs/1505.04597)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_segmentation/unet_cub200/unet_cub200.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_segmentation/unet_cub200/unet_cub200.ipynb)]

* Image Style Transfer
    * Fast Style Transfer: Altering the source image style to a target art style using MSCOCO 2017 dataset. [[paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_styletransfer/fst_coco/fst_coco.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_styletransfer/fst_coco/fst_coco.ipynb)]

* Tabular data
    * DNN: Predicting house price with deep neural network. [[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn_housing/dnn_housing.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn_housing/dnn_housing.ipynb)]

## Contribution
If you are looking for some implementations that we haven't done yet and want to join our efforts of making state-of-the-art AI easier, please consider contribute to us and we would really appreciate it! :smiley: