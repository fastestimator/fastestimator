# FastEstimator

[![Build Status](http://54.184.62.55:8080/buildStatus/icon?job=fe_git%2Ffastestimator%2Fmaster)](http://54.184.62.55:8080/job/fe_git/job/fastestimator/job/master/)

FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:

## Prerequisites:
* Python3
* TensorFlow2

    * GPU:  `pip install tensorflow-gpu==2.0.0-rc1`
    * CPU:  `pip install tensorflow==2.0.0-rc1`


## Installation
`pip install fastestimator==1.0a0`

## [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags)
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator environment, users can pull image from Docker Hub.

* GPU: `docker pull fastestimator/fastestimator:1.0a0-gpu`
* CPU: `docker pull fastestimator/fastestimator:1.0a0-cpu`

## Start your first FastEstimator training

```
$ python ./apphub/image_classification/lenet_mnist.py
```

## Documentation
For more info about building models and training, check  out [FastEstimator documentation](https://fastestimator.org)

## License
[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)
