# FastEstimator Application Hub

Welcome to FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art accessible to everyone.

## Purpose of Application Hub

* Provide place to learn implementation details of state-of-the-art
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own project/product


## Why not just learn from official implementation
If you ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000+ lines of code or 500+ lines of code to implement the same idea using different AI frameworks. In FastEstimator, we strive to make things simpler and more intuitive while preserving the flexibility. As a result, many state-of-the-art AI implementations can be simplified greatly such that the code directly reflects the ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) include 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_tf.py) only uses 500+ lines.

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
* **Text Classification:** Sentiment classification based on IMDB movie review. [[paper](https://www.bioinf.jku.at/publications/older/2604.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb.ipynb)]

* **Named Entity Recognition**: BERT fine tuning on German news corpora. [[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert.ipynb)]

### Adversarial Training
* **Fast Gradient Sign Method:** Adversarial training on CIFAR-10 dataset using Fast Gradient Sign Method [[paper](https://arxiv.org/abs/1412.6572)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm.ipynb)]

### Image Classification
* **CIFAR-10 Fast:** Cifar-10 Image Classification using ResNet [[paper](https://arxiv.org/abs/1608.06993)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast.ipynb)]

* **MNIST:** Convolutional Neural Networks trained on Mnist dataset using LeNet. [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist.ipynb)]

### Image Generation
* **CVAE:** Image feature representation learning and image generation with Convolutional Variational AutoEncoder on Mnist dataset. [[paper](https://arxiv.org/abs/1312.6114)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae.ipynb)]

* **DCGAN:** Image generation with Deep Convolutional GAN on Mnist dataset. [[paper](https://arxiv.org/abs/1511.06434)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan.ipynb)]

* **PGGAN:** High-resolution image generation with progressive growing of GANs on NIH chest X-ray dataset. [[paper](https://arxiv.org/abs/1710.10196)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan.ipynb)]

### Semantic Segmentation
* **UNet:** Lung segmentation with UNet using chest X-ray dataset. [[paper](https://arxiv.org/abs/1505.04597)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet.ipynb)]

### Instance Detection
* **RetinaNet:** object detection with RetinaNet on COCO2017 dataset. [[paper](https://arxiv.org/abs/1708.02002)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/instance_detection/retinanet/retinanet_tf.py)]

### Image Style Transfer
* **Fast Style Transfer:** Altering the source image style to a target art style using MSCOCO 2017 dataset. [[paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst.ipynb)]

### Tabular Data
* **DNN:** Classifying breast tumors as benign or malignant with a deep neural network. [[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn.ipynb)]

### One-shot Learning
* **Siamese Network:** Alphabet character recognition using Siamese network. [[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese.ipynb)]

### Multi-task Learning
* **Uncertainty Weighted Loss:** Train Uncertainty Network that learns to dynamically adjust the task-related weighted loss from uncertainty. [[paper](https://arxiv.org/abs/1705.07115)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss.ipynb)]

## Contributions
If you have implementations that we haven't done yet and want to join our efforts of making state-of-the-art AI easier, please consider contribute to us. We would really appreciate it! :smiley:
