============
Installation
============

Prerequisites
-------------
- Python3
- TensorFlow 1.12.0
- Horovod (Only needed for distributed training)

Installation
----------------

  #. ``pip install fastestimator``

Docker
-------

Docker container creates isolated virtual environment that shares resources with host machine. 
Docker provides an easy way to set up FastEstimator running environment, users can either build 
image from dockerfile or pull image from Docker-Hub_.

Current docker image only provides FastEstimator dependencies, users need to install FastEstimator inside the container until opensource.

Build Image from docker file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:GPU: ``docker build -t fastestimator -f docker/Dockerfile.gpu``

:CPU: ``docker build -t fastestimator -f docker/Dockerfile.cpu``

Pull Image from Docker Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:GPU: ``docker pull fastestimator/fastestimator:latest-gpu``

.. _Docker-Hub: https://hub.docker.com/r/fastestimator/fastestimator/tags
