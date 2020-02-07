# FastEstimator

[![Build Status](http://52.36.103.172:8080/buildStatus/icon?subject=PR-build&job=fe_git%2Ffastestimator%2Fmaster)](http://52.36.103.172:8080/job/fe_git/job/fastestimator/job/master/)
[![Build Status](http://52.36.103.172:8080/buildStatus/icon?subject=nightly-build&job=nightly)](http://52.36.103.172:8080/job/nightly/)

FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:


## Prerequisites:
* Python >= 3.5
* TensorFlow2

    * GPU:  `pip install tensorflow-gpu==2.0.0`
    * CPU:  `pip install tensorflow==2.0.0`

* Pytorch backend is coming soon!

## Installation
Please choose one:
* I have no idea what FastEstimator is about:
```
pip install fastestimator==1.0b2
```
* I want to keep up to date with the latest:
```
pip install fastestimator-nightly
```
* I'm here to play hardcore mode:

```
git clone https://github.com/fastestimator/fastestimator.git
pip install -e fastestimator
```


## Docker Hub
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator environment, users can pull image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags).

* GPU: `docker pull fastestimator/fastestimator:1.0b2-gpu`
* CPU: `docker pull fastestimator/fastestimator:1.0b2-cpu`

## Start your first FastEstimator training

```
$ python ./apphub/image_classification/lenet_mnist/lenet_mnist.py
```

## Tutorial

We have [tutorial series](https://github.com/fastestimator/fastestimator/tree/master/tutorial) that walk through everything you need to know about FastEstimator.

## Example

Check out [Application Hub](https://github.com/fastestimator/fastestimator/tree/master/apphub) for end-to-end deep learning examples in FastEstimator.

## Documentation
For more info, check out our [FastEstimator documentation](https://www.fastestimator.org).

## Citation
Please cite FastEstimator in your publications if it helps your research:
```
@misc{dong2019fastestimator,
    title={FastEstimator: A Deep Learning Library for Fast Prototyping and Productization},
    author={Xiaomeng Dong and Junpyo Hong and Hsi-Ming Chang and Michael Potter and Aritra Chowdhury and
            Purujit Bahl and Vivek Soni and Yun-Chan Tsai and Rajesh Tamada and Gaurav Kumar and Caroline Favart and
            V. Ratna Saripalli and Gopal Avinash},
    year={2019},
    eprint={1910.04875},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License
[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)