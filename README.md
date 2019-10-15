# FastEstimator

[![Build Status](http://54.184.62.55:8080/buildStatus/icon?job=fe_git%2Ffastestimator%2Fmaster)](http://54.184.62.55:8080/job/fe_git/job/fastestimator/job/master/)
[![Build Status](http://54.184.62.55:8080/buildStatus/icon?subject=build-nightly&job=fastestimator-nightly)](http://54.184.62.55:8080/job/fastestimator-nightly/)

FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:

## Prerequisites:
* Python >= 3.5
* TensorFlow2

    * GPU:  `pip install tensorflow-gpu==2.0.0`
    * CPU:  `pip install tensorflow==2.0.0`


## Installation
`pip install fastestimator==1.0b0`

## [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags)
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator environment, users can pull image from Docker Hub.

* GPU: `docker pull fastestimator/fastestimator:1.0b0-gpu`
* CPU: `docker pull fastestimator/fastestimator:1.0b0-cpu`

## Start your first FastEstimator training

```
$ python ./apphub/image_classification/lenet_mnist/lenet_mnist.py
```

## Documentation
For more info, check out [FastEstimator documentation](https://fastestimator.org)

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

