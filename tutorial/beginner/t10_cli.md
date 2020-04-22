# Tutorial 10: How to use FastEstimator CLI

## Overview
FastEstimator comes with a set of CLI tools that can help users train and test their models quickly. In this tutorial, we will go through the CLI usage and the arguments the CLI commands  take. This tutorial is divided into the following sections:

* How does the CLI work
* CLI usage
* Sending input args to `get_estimator`


## How does the CLI work
The CLI takes a command followed by the entry point and optional arguments. The entry point is a path to the python file that contains the estimator definition, which is declared in the `get_estimator()` function. This function returns the estimator which is later used to call either the `fit()` or `test()` functions used to train or test the model respectively.

## CLI usage
In this section we will show the actual commands that we can use to train and test our models. The following snippet shows the command used to train our apphub MNIST example:

`fastestimator train apphub/image_classification/mnist/mnist_tf.py`

To evaluate our MNIST example we can use the followowing:

`fastestimator test apphub/image_classification/mnist/mnist_tf.py`

## Sending input args to `get_estimator`
As we mentioned above, we can also pass arguments using the CLI. The two main ways you can do that is either by explicitly specifying the arguments or by passing a json configuration containing the details about the arguments.
* To pass the arguments directly we can use the `--arg` format. The following shows an example of how we can set the number of epochs in our MNIST example above to 3:

    `fastestimator train apphub/image_classification/mnist/mnist_tf.py --epochs 3`

* The other way we talked about was by using the `--hyperparameters` argument and by passing it a json file containing all the training hyperparameters like epochs, batch_size, optimizer, etc. This option is really useful when you want to repeat the training job more than once and/or the list of the hyperparameter is getting really long. The following shows a json and how it can be used for our MNIST example:
    ```
    JSON:
    {
        "epochs": 1,
        "batch_size": 64
    }
    ```
    `fastestimator train apphub/image_classification/mnist/mnist_tf.py --hyperparameters hp.json`