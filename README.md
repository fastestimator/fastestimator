# FastEstimator

[![License](https://img.shields.io/badge/License-Apache_2.0-informational.svg)](LICENSE)
[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=PR-build&job=fastestimator%2Ffastestimator%2Fmaster)](http://jenkins.fastestimator.org:8080/job/fastestimator/job/fastestimator/job/master/)
[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=nightly-build&job=nightly)](http://jenkins.fastestimator.org:8080/job/nightly/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3a46ea86b8f04caab271f2a7bd6f4bd9)](https://www.codacy.com/gh/fastestimator/fastestimator/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fastestimator/fastestimator&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/3a46ea86b8f04caab271f2a7bd6f4bd9)](https://www.codacy.com/gh/fastestimator/fastestimator/dashboard?utm_source=github.com&utm_medium=referral&utm_content=fastestimator/fastestimator&utm_campaign=Badge_Coverage)
[![PyPI version](https://badge.fury.io/py/fastestimator.svg)](https://pypi.org/project/fastestimator/)
[![PyPI stable Download](https://img.shields.io/pypi/dm/fastestimator?label=stable%20downloads&color=16D1B4)](https://pypistats.org/packages/fastestimator)
[![PyPI stable Download](https://img.shields.io/pypi/dm/fastestimator-nightly?label=nightly%20downloads&color=16D1B4)](https://pypistats.org/packages/fastestimator-nightly)


FastEstimator is a high-level deep learning library built on TensorFlow2 and PyTorch. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:

## Prerequisites:
* Python >= 3.6
* TensorFlow == 2.4.1
* PyTorch == 1.7.1

## Installation:
### 1. Install Dependencies:

* Install TensorFlow
    ```bash
    pip install tensorflow==2.4.1
    ```

* Install PyTorch
    * CPU:
        ```bash
        pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
        ```

    * GPU:
        ```bash
        pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
        ```
* Extra Dependencies:

    * Windows:

        * Install Build Tools for Visual Studio 2019 [here](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).

        * Install latest Visual C++ redistributable [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and choose x86 for 32 bit OS, x64 for 64 bit OS.

    * Linux:
        ``` bash
        $ apt-get install libglib2.0-0 libsm6 libxrender1 libxext6
        ```

    * Mac:
        ``` bash
        $ echo No extra dependency needed ":)"
        ```

### 2. Install FastEstimator:
* Stable (Linux/Mac):
    ``` bash
    $ pip install fastestimator
    ```

* Stable (Windows):

    First download zip file from [available releases](https://github.com/fastestimator/fastestimator/releases)
    ``` bash
    $ pip install fastestimator-x.x.x.zip
    ```

* Nightly (Linux/Mac):
    ``` bash
    $ pip install fastestimator-nightly
    ```

* Nightly (Windows):

    First download zip file [here](https://github.com/fastestimator/fastestimator/archive/master.zip)
    ``` bash
    $ pip install fastestimator-master.zip
    ```



## Docker Hub
Docker containers create isolated virtual environments that share resources with a host machine. Docker provides an easy way to set up a FastEstimator environment. You can simply pull our image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags) and get started:
* Stable:
    * GPU:
        ``` bash
        docker pull fastestimator/fastestimator:latest-gpu
        ```
    * CPU:
        ``` bash
        docker pull fastestimator/fastestimator:latest-cpu
        ```
* Nighly:
    * GPU:
        ``` bash
        docker pull fastestimator/fastestimator:nightly-gpu
        ```
    * CPU:
        ``` bash
        docker pull fastestimator/fastestimator:nightly-cpu
        ```

## Useful Links:
* [Website](https://www.fastestimator.org): More info about FastEstimator API and news.
* [Tutorial Series](https://github.com/fastestimator/fastestimator/tree/master/tutorial): Everything you need to know about FastEstimator.
* [Application Hub](https://github.com/fastestimator/fastestimator/tree/master/apphub): End-to-end deep learning examples in FastEstimator.



## Citation
Please cite FastEstimator in your publications if it helps your research:
```
@misc{fastestimator,
  title  = {FastEstimator: A Deep Learning Library for Fast Prototyping and Productization},
  author = {Xiaomeng Dong and Junpyo Hong and Hsi-Ming Chang and Michael Potter and Aritra Chowdhury and
            Purujit Bahl and Vivek Soni and Yun-Chan Tsai and Rajesh Tamada and Gaurav Kumar and Caroline Favart and
            V. Ratna Saripalli and Gopal Avinash},
  note   = {NeurIPS Systems for ML Workshop},
  year   = {2019},
  url    = {http://learningsys.org/neurips19/assets/papers/10_CameraReadySubmission_FastEstimator_final_camera.pdf}
}
```

## License
[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)
