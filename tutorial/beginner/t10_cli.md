# Tutorial 10: How to use FastEstimator Command Line Interface (CLI)

## Overview
FastEstimator comes with a set of CLI commands that can help users train and test their models quickly. In this tutorial, we will go through the CLI usage and the arguments these CLI commands take. This tutorial is divided into the following sections:

* [How Does the CLI Work](#t10intro)
* [CLI Usage](#t10usage)
* [Sending Input Args to `get_estimator`](#t10args)
    * [Using --arg](#t10arg)
    * [Using a JSON file](#t10json)

<a id='t10intro'></a>
## How Does the CLI Work
Given a python file, the FastEstimator CLI looks for a `get_estimator()` function to get the estimator definition. It then calls either the `fit()` or `test()` functions on the returned estimator instance to train or test the model.

<a id='t10usage'></a>
## CLI Usage
In this section we will show the actual commands that we can use to train and test our models. To start the training we use the following:

`fastestimator train apphub/image_classification/mnist/mnist_tf.py`

To test our trained model we use the following:

`fastestimator test apphub/image_classification/mnist/mnist_tf.py`

<a id='t10args'></a>
## Sending Input Args to `get_estimator`
We can also pass arguments to the `get_estimator()` call from the CLI. The following code snippet shows the `get_estimator()` method for our MNIST example:
```python
def get_estimator(epochs=2, batch_size=32, ...):
    ...
```

Next, we try to change two of these arguments:

<a id='t10arg'></a>
### Using --arg
To pass the arguments directly from the CLI we can use the `--arg` format. The following shows an example of how we can set the number of epochs to 3 and batch_size to 64:

`fastestimator train apphub/image_classification/mnist/mnist_tf.py --epochs 3 --batch_size 64`

<a id='t10json'></a>
### Using a JSON file
The other way we can send arguments is by using the `--hyperparameters` argument and passing it a json file containing all the training hyperparameters like epochs, batch_size, optimizer, etc. This option is really useful when you want to repeat the training job more than once and/or the list of the hyperparameter is getting really long. The following shows an example JSON file and how it could be used for our MNIST example:
```
JSON:
{
    "epochs": 1,
    "batch_size": 64
}
```
`fastestimator train apphub/image_classification/mnist/mnist_tf.py --hyperparameters hp.json`