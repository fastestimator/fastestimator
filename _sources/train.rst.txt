How to train with FastEstimator
===============================

Users can choose one of two ways to start training. Here's the general rule of thumb: 

* Use Command Line Interface for:
  
  * General purpose training
  * Distributed training
  * Hyperparameter tuning

* Use Python Console for:

  * Debugging

Next We use Mnist data as example to demonstrate various ways to launch training jobs. 


Command Line Interface
----------------------

`fastestimator` command will be available after installation, in order to use the command, 
users should prepare entry point -- python script that defines the `get_estimator` function. Here's the usage:

.. code-block:: bash

    fastestimator train  

    Required arguments:
    
    --entry_point:      Path of python script that defines 'get_estimator' 

    Optional arguments:

    --input:            Directory where tfrecord is saved or will be saved 
                        (default: /tmp/FEdataset)
    --hyperparameters:  Path of JSON file that defines the input arguments of 
                        'get_estimator' in a dictionary format (default: None)
    --num_process:      Number of parallel training process (default: 1)
    --arg1:             Input argument1 of `get_estimator` function (default: None)
    ...                 ...
    --argN:             Input argumentN of `get_estimator` function (default: None)

Example 1: Using entry point only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, `get_estimator` takes default argument or no argument (see `mnist`_). 
This format is best suited for running a quick single experiment.

.. code-block:: bash

    fastestimator train --entry_point mnist.py 

if you want to save tfrecord to a specific direcoty (E.g, /data/mnist):

.. code-block:: bash

    fastestimator train --entry_point mnist.py --input /data/mnist

Example 2: Using entry point with arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, user can pass arguments of `get_estimator` from command line. This is useful when experimenting different parameters.

.. code-block:: bash

    fastestimator train \
    --entry_point mnist.py \
    --input /data/mnist \
    --epochs 3 \
    --batch_size 64 \
    --optimizer sgd

Example 3: Using JSON file as arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It may be cool to pass two or three arguments on terminal, but when the number of arguments is too 
big (say ~100), it doesn't seem so cool anymore. Alternatively, you pass a `JSON file`_ that contains all arguments of `get_estimator` in a dictionary format.

.. code-block:: bash

    python -m fastestimator.fit \
    --entry_point mnist.py \
    --input /data/mnist \
    --hyperparameters mnist_args.json

Python Console
--------------

Example 4: Train in python console
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users that prefer running codes in python console (e.g., Jupyter notebook, Spyder, Pycharm), check out `this notebook`_ for python console training.

.. code-block:: python

    def get_estimator(...)
    ...
    return estimator

    estimator = get_estimator(...)
    estimator.fit("/data/mnist")



.. _mnist: https://github.com/fastestimator/fastestimator/blob/r0.2/tutorial/mnist.py
.. _JSON file: https://github.com/fastestimator/fastestimator/blob/r0.2/tutorial/mnist_args.json
.. _this notebook: https://github.com/fastestimator/fastestimator/blob/r0.2/tutorial/mnist.ipynb

