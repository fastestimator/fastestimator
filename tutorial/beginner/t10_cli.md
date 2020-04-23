# Tutorial 10: How to use FastEstimator Command Line Interface (CLI)

## Overview
CLI is a command line program that accepts text input to execute the program's functions. FastEstimator comes with a set of CLI commands that can help users train and test their models quickly. In this tutorial, we will go through the CLI usage and the arguments these CLI commands take. This tutorial is divided into the following sections:

* How does the CLI work
* CLI usage
* Sending input args to `get_estimator`


## How does the CLI work
Given a python file, the FastEstimator CLI looks for `get_estimator()` function to get the estimator definition, then we call either the `fit()` or `test()` functions on it to train or test the model.

## CLI usage
In this section we will show the actual commands that we can use to train and test our models. To start the training we use the following:

`fastestimator train apphub/image_classification/mnist/mnist_tf.py`

To test our trained model we use the following:

`fastestimator test apphub/image_classification/mnist/mnist_tf.py`

## Sending input args to `get_estimator`
We can also pass arguments that the `get_estimator()` takes to the CLI. The following code snippet shows which arguments our MNIST example takes:
```python
def get_estimator(epochs=2, batch_size=32, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    ...
```

Next, we try to change two of these arguments using the `--arg` and JSON file format.

### Using --arg
To pass the arguments directly from the CLI we can use the `--arg` format. The following shows an example of how we can set the number of epochs to 3 and batch_size to 64:

`fastestimator train apphub/image_classification/mnist/mnist_tf.py --epochs 3 --batch_size 64`

### Using JSON file
The other way we can send arguments is by using the `--hyperparameters` argument and by passing it a json file containing all the training hyperparameters like epochs, batch_size, optimizer, etc. This option is really useful when you want to repeat the training job more than once and/or the list of the hyperparameter is getting really long. The following shows a json and how it can be used for our MNIST example:
```
JSON:
{
    "epochs": 1,
    "batch_size": 64
}
```
`fastestimator train apphub/image_classification/mnist/mnist_tf.py --hyperparameters hp.json`