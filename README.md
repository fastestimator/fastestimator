# FastEstimator 

[![Build Status](http://54.184.62.55:8080/buildStatus/icon?job=fe_git%2Ffastestimator%2Fmaster)](http://54.184.62.55:8080/job/fe_git/job/fastestimator/job/master/)

FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:

## Prerequisites:
* Python3
* TensorFlow 2.0 beta1
* Horovod (Only needed for distributed training)

## Installation
`pip install fastestimator`

## Docker
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator running environment, users can either build from dockerfile or pull from Docker Hub.

### Build Image from Dockerfile
* GPU: `docker build -t fastestimator_gpu -f docker/Dockerfile.gpu .`
* CPU: `docker build -t fastestimator_cpu -f docker/Dockerfile.cpu .`

### Pull Image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags)
* GPU: `docker pull fastestimator/fastestimator:latest-gpu`
* CPU: `docker pull fastestimator/fastestimator:latest-cpu`

## Running your first FastEstimator training
* macOS/Linux:
```
$ fastestimator train --entry_point tutorial/mnist.py
```
* Windows, macOS and Linux:

    check out this [notebook](https://github.com/fastestimator/fastestimator/blob/r0.2/tutorial/mnist.ipynb)

## Tests
```
$ ./fastestimator/test/tests.sh
```

## Documentation
For more info about building models and training, check  out [FastEstimator documentation](https://fastestimator.org)

## License
[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)
