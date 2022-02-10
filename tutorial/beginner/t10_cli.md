# Tutorial 10: How to use FastEstimator Command Line Interface (CLI)

## Overview
FastEstimator comes with a set of CLI commands that can help users train and test their models quickly. In this tutorial, we will go through the CLI usage and the arguments these CLI commands take. This tutorial is divided into the following sections:

* [How Does the CLI Work](#t10intro)
* [CLI Usage](#t10usage)
* [Sending Input Args to `get_estimator`](#t10args)
  * [Using --arg](#t10arg)
  * [Using a JSON file](#t10json)
  * [System argument](#t10system)

<a id='t10intro'></a>
## How Does the CLI Work
* `fastestimator train`: the command will look for a `get_estimator` function, invoke it, and then call the `fit()` method on the returned estimator instance to start the training.
* `fastestimator test`: the command will look for a `get_estimator` function, invoke it, and then call the `test()` method on the returned estimator instance to run testing.
* `fastestimator run`: the command will look for a `fastestimator_run` function and invoke it. If `fastestimator_run` is not available, it will instead look for `get_estimator`, invoke it, and then call `fit()` and/ or `test` depending on what data is available within the estimator's Pipeline.

###

<a id='t10usage'></a>
## CLI Usage
In this section we will show the actual commands that we can use to train and test our models. We will use [mnist_tf.py](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_classification/mnist/mnist_tf.py) for illustration.

To call `estimator.fit()` and start the training on terminal:

``` bash
$ fastestimator train mnist_tf.py
```

To call `estimator.test()` and start testing on terminal:

``` bash
$ fastestimator test mnist_tf.py
```

To first call `estimator.fit()` then `estimator.test()`, you can use:
``` bash
$ fastestimator run mnist_tf.py
```

<a id='t10args'></a>
## Sending Input Args to `get_estimator` or `fastestimator_run`
We can also pass arguments to the `get_estimator` or `fastestimator_run` functions from the CLI. The following code snippet shows the `get_estimator` method for our MNIST example:
```python
def get_estimator(epochs=2, batch_size=32, ...):
    ...
```

Next, we try to change these arguments in two ways:

<a id='t10arg'></a>
### Using --arg
To pass the arguments directly from the CLI we can use the `--arg` format. The following shows an example of how we can set the number of epochs to 3 and batch_size to 64:

``` bash
$ fastestimator train mnist_tf.py --epochs 3 --batch_size 64
```

<a id='t10json'></a>
### Using a JSON file
The other way we can send arguments is by using the `--hyperparameters` argument and passing it a json file containing all the training hyperparameters like epochs, batch_size, optimizer, etc. This option is really useful when you want to repeat the training job more than once and/or the list of the hyperparameter is getting really long. The following shows an example JSON file and how it could be used for our MNIST example:
``` json
{
    "epochs": 1,
    "batch_size": 64
}
```
``` bash
$ fastestimator train mnist_tf.py --hyperparameters hp.json
```
<a id='t10system'></a>
### System argument
There are some default system arguments in the `fastestimator train` and `fastestimator test` commands. Here are a list of them:
* `warmup`:  Only available in `fastestimator train`, it controls whether to perform warmup checking before the actual training starts. Default is True. Users can disable warmup before training by `--warmup False`.
* `summary`: Available in both `fastestimator train` and `fastestimator test`, this is the same argument used in `estimator.fit()` or `estimator.test()`. It allows users to specify experiment name when generating reports. For example, Users can set experiment name by `--summary exp_name`.
