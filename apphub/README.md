# FastEstimator Application Hub

Welcome to FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art accessible to everyone.

## Purpose of Application Hub

* Provide place to learn implementation details of state-of-the-art
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own project/product


## Why not just learn from official implementation
If you ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000+ lines of code or 500+ lines of code to implement the same idea using different AI frameworks. In FastEstimator, we strive to make things simpler and more intuitive while preserving the flexibility. As a result, many state-of-the-art AI implementations can be simplified greatly such that the code directly reflects the ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) include 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/tree/master/apphub/image_generation/pggan_nihchestxray) only uses 500+ lines.

To summarize, we spent time learning from the official implementation, so you can save time by learning from us!

## What's included in each example

Each example contains two files:

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
### Natural Language Processing (NLP)
* **LSTM:** Sentimental classification based on IMDB movie review. [[paper](https://www.bioinf.jku.at/publications/older/2604.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/lstm_imdb/lstm_imdb.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/lstm_imdb/lstm_imdb.ipynb)]

### Domain Adaptation
* **Adversarial Discriminative Domain Adaptation (ADDA):** Adapting trained MNIST clasifier to USPS digits dataset without label data. [[paper](https://arxiv.org/abs/1702.05464)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/domain_adaptation/ADDA/ADDA.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/domain_adaptation/ADDA/ADDA.ipynb)]

* **Unsupervised Domain Adaptation by Backpropagation:** Training a Domain Adversarial Neural Network (DANN) on MNIST dataset  (with label) and USPS dataset (without label) via gradient reversal layer. [[paper](http://proceedings.mlr.press/v37/ganin15.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/domain_adaptation/DANN/DANN.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/domain_adaptation/DANN/DANN.ipynb)]

### Image Classification
* **DenseNet:** Image classifier with DenseNet121 on Cifar10 dataset. [[paper](https://arxiv.org/abs/1608.06993)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/densenet121_cifar10/densenet121_cifar10.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/densenet121_cifar10/densenet121_cifar10.ipynb)]

* **Adversarial Attack:** Image classifier training to resist adversarial attaks on Cifar10 dataset. [[paper](https://arxiv.org/abs/1412.6572)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_adversarial/lenet_cifar10_adversarial.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_adversarial/lenet_cifar10_adversarial.ipynb)]

* **Mixup:** Image classifier training that uses mix up as data augmentation strategy to enhance robustness on Cifar10 dataset. [[paper](https://arxiv.org/abs/1710.09412)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_mixup/lenet_cifar10_mixup.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_cifar10_mixup/lenet_cifar10_mixup.ipynb)]

* **LeNet:** Convolutional Neural Networks trained on Mnist dataset. [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_mnist/lenet_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/lenet_mnist/lenet_mnist.ipynb)]

### Image Generation
* **CVAE:** Image feature representation learning and image generation with Convolutional Variational AutoEncoder on Mnist dataset. [[paper](https://arxiv.org/abs/1312.6114)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae_mnist/cvae_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae_mnist/cvae_mnist.ipynb)]

* **Cycle-GAN:** Unpaired image-to-image translation between two domain using Cycle-GAN.[[paper](https://arxiv.org/abs/1703.10593)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan_horse2zebra/cyclegan_horse2zebra.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan_horse2zebra/cyclegan_horse2zebra.ipynb)]

* **DCGAN:** Image generation with Deep Convolutional GAN on Mnist dataset. [[paper](https://arxiv.org/abs/1511.06434)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan_mnist/dcgan_mnist.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan_mnist/dcgan_mnist.ipynb)]

* **PGGAN:** High-resolution image generation with progressive growing of GANs on NIH chest X-ray dataset. [[paper](https://arxiv.org/abs/1710.10196)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan_nihchestxray/pggan_nihchestxray_128.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan_nihchestxray/pggan_nihchestxray_128.ipynb)]

### Semantic Segmentation
* **UNet:** Lung segmentation with UNet using chest X-ray dataset. [[paper](https://arxiv.org/abs/1505.04597)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet_montgomery/unet_montgomery.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet_montgomery/unet_montgomery.ipynb)]

### Object Detection
* **RetinaNet:** object detection with RetinaNet on COCO2017 dataset. [[paper](https://arxiv.org/abs/1708.02002)][[code](https://github.com/vbvg2008/fastestimator/blob/master/apphub/instance_detection/retinanet_coco/retinanet_coco.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/instance_detection/retinanet_coco/retinanet_coco.ipynb)]

### Image Style Transfer
* **Fast Style Transfer:** Altering the source image style to a target art style using MSCOCO 2017 dataset. [[paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_coco.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_coco.ipynb)]

### Tabular Data
* **DNN:** Classifying breast tumors as benign or malignant with a deep neural network. [[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn_breast_cancer/dnn_breast_cancer.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn_breast_cancer/dnn_breast_cancer.ipynb)]

### One-shot Learning
* **Siamese Network:** Alphabet character recognition using Siamese network. [[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_network.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_network.ipynb)]

### Multi-task Learning
* **Uncertainty Weighted Loss:** Train Uncertainty Network that learns to dynamically adjust the task-related weighted loss from uncertainty. [[paper](https://arxiv.org/abs/1705.07115)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_cub200.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_cub200.ipynb)]

### Meta-Learning
* **Model-Agnostic Meta-Learning:** Train a meta model for a 10-shot sinusoid regression. [[paper](https://arxiv.org/abs/1703.03400)][[code](https://github.com/fastestimator/fastestimator/blob/master/apphub/meta_learning/MAML/maml.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/meta_learning/MAML/maml.ipynb)]

## Contributions
If you have implementations that we haven't done yet and want to join our efforts of making state-of-the-art AI easier, please consider contribute to us. We would really appreciate it! :smiley:
