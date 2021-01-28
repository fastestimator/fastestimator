# FastEstimator Application Hub

Welcome to the FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art solutions accessible to everyone.

## Purpose of Application Hub

* Provide a place to learn implementation details of state-of-the-art solutions
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own projects/products

## Why not just learn from official implementations?

If you have ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000 lines of code to implement an idea which could have been expressed in 500 lines using a different AI framework. In FastEstimator, we strive to make things simpler and more intuitive while preserving flexibility. As a result, many state-of-the-art AI implementations can be simplified greatly such that the code directly reflects the key ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) includes 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_tf.py) requires less than 500.

To summarize, we spent time learning from the official implementation, so you can save time by learning from us!

## What's included in each example?

Each example contains three files:

1. A TensorFlow python file (.py): The FastEstimator source code needed to run the example with TensorFlow.
2. A PyTorch python file (.py): The FastEstimator source code needed to run the example with PyTorch.
3. A jupyter notebook (.ipynb): A notebook that provides step-by-step instructions and explanations about the implementation.

## How do I run each example

One can simply execute the python file of any example:
``` bash
$ python mnist_tf.py
```

Or use our Command-Line Interface (CLI):

``` bash
$ fastestimator train mnist_torch.py
```

One benefit of the CLI is that it allows users to configure the input args of `get_estimator`:

``` bash
$ fastestimator train lenet_mnist.py --batch_size 64 --epochs 4
```

## Table of Contents:
### Natural Language Processing (NLP)
* **Text Classification:** Sentiment classification based on IMDB movie reviews. [[paper](https://www.bioinf.jku.at/publications/older/2604.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/imdb/imdb.ipynb)]

* **Named Entity Recognition**: BERT fine tuning on a German news corpora. [[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/named_entity_recognition/bert.ipynb)]

* **Language Modeling:** Word level language model on Penn Treebank dataset. [[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/language_modeling/ptb_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/language_modeling/ptb_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/NLP/language_modeling/ptb.ipynb)]

### Adversarial Training
* **Fast Gradient Sign Method:** Adversarial training on the ciFAIR-10 dataset using the Fast Gradient Sign Method. [[paper](https://arxiv.org/abs/1412.6572)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/fgsm/fgsm.ipynb)]

* **Adversarial Robustness with Error Correcting Codes:** [[paper](https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-robustness-of-deep-neural-networks.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc/ecc_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc/ecc_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc/ecc.ipynb)]

* **Adversarial Robustness with Error Correcting Codes (and Hinge Loss):** [[paper](https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-robustness-of-deep-neural-networks.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc_hinge/ecc_hinge_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc_hinge/ecc_hinge_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/adversarial_training/ecc_hinge/ecc_hinge.ipynb)]

### Anomaly Detection
* **Adversarially Learned One-Class Classifier for Novelty Detection:** Adversarial training to detect anomalies using MNIST dataset. [[paper](https://arxiv.org/pdf/1802.09088v2.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/anomaly_detection/alocc/alocc_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/anomaly_detection/alocc/alocc_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/anomaly_detection/alocc/alocc.ipynb)]

### AutoML
* **RandAugment**: Practical automated data augmentation with a reduced search space. [[paper](https://arxiv.org/abs/1909.13719)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/automl/rand_augment/rand_augment_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/automl/rand_augment/rand_augment_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/automl/rand_augment/rand_augment.ipynb)]


### Image Classification
* **CIFAR-10 Fast:** ciFAIR-10 Image Classification using ResNet. [[paper](https://arxiv.org/abs/1608.06993)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/cifar10_fast/cifar10_fast.ipynb)]

* **MNIST:** Convolutional Neural Networks trained on MNIST using LeNet. [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist.ipynb)]

* **Pyramid Network:** Deep Pyramidal Residual Networks. [[paper](https://arxiv.org/abs/1610.02915)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/pyramidnet/pyramidnet_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/pyramidnet/pyramidnet_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/pyramidnet/pyramidnet.ipynb)]

### Image Generation
* **CVAE:** Image feature representation learning and image generation with a Convolutional Variational AutoEncoder on the MNIST dataset. [[paper](https://arxiv.org/abs/1312.6114)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cvae/cvae.ipynb)]

* **Cycle-GAN:** Unpaired image-to-image translation between two domains using Cycle-GAN. [[paper](https://arxiv.org/abs/1703.10593)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan/cyclegan_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan/cyclegan_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/cyclegan/cyclegan.ipynb)]

* **DCGAN:** Image generation with a Deep Convolutional GAN on the MNIST dataset. [[paper](https://arxiv.org/abs/1511.06434)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/dcgan/dcgan.ipynb)]

* **PGGAN:** High-resolution image generation with progressive growing of GANs on an NIH chest X-ray dataset. [[paper](https://arxiv.org/abs/1710.10196)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan.ipynb)]

### Semantic Segmentation
* **UNet:** Lung segmentation with UNet using a chest X-ray dataset. [[paper](https://arxiv.org/abs/1505.04597)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/semantic_segmentation/unet/unet.ipynb)]

### Instance Detection
* **RetinaNet:** Object detection with RetinaNet on the COCO2017 dataset. [[paper](https://arxiv.org/abs/1708.02002)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/instance_detection/retinanet/retinanet_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/instance_detection/retinanet/retinanet_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/instance_detection/retinanet/retinanet.ipynb)]

### Image Style Transfer
* **Fast Style Transfer:** Altering the source image style to a target art style using the MSCOCO 2017 dataset. [[paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/style_transfer/fst_coco/fst.ipynb)]

### Tabular Data
* **DNN:** Classifying breast tumors as benign or malignant with a deep neural network. [[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/tabular/dnn/dnn.ipynb)]

### One-shot Learning
* **Siamese Network:** Alphabet character recognition using a Siamese network. [[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/one_shot_learning/siamese.ipynb)]

### Multi-task Learning
* **Uncertainty Weighted Loss:** Train an Uncertainty Network that learns to dynamically adjust the task-related weighted loss using uncertainty metrics. [[paper](https://arxiv.org/abs/1705.07115)][[tensorflow code](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_tf.py)][[pytorch code](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss_torch.py)][[notebook](https://github.com/fastestimator/fastestimator/blob/master/apphub/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss.ipynb)]

## Contributions
If you have implementations that we haven't done yet and want to join our efforts to make state-of-the-art AI easier, please consider contributing to our repo. We would really appreciate it!
