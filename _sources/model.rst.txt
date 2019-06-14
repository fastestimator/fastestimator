Buiding model with FastEstimator
======================================
If you ever go to a zoo, it may not be surprising to see that animals all have 3 parts: 
head, body and leg. Among different animals, no matter how much their appearances may vary, 
the differences are nothing but different arrangements of the 3 common parts. 

Similarly, in the deep learning model zoo, every model can be described in 3 components: model architecture,
data pipeline and training strategy. Each component of the model serves its unique purpose and functionality:

* Model Architecture: stores trainable & differentiable operations.
* Data Pipeline: performs series of preprocess operations and transports data from disk/RAM to model.
* Training Strategy: iterates data pipeline and model architecture in an optimization process.

Each of the component above represents a critical building block of FastEstimator and the story begins with them.

Overview
-----------
There are 3 main API components that users will be interacting with: ``Pipeline``, 
``Network`` and ``Estimator``. The workflow of FastEstimator is a 3-step process as shown below 
in the Mnist example.

.. code-block:: python
    
    from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape
    from fastestimator.estimator.estimator import Estimator
    from fastestimator.pipeline.pipeline import Pipeline
    from fastestimator.network.network import Network
    from fastestimator.application.lenet import LeNet
    import tensorflow as tf

    def get_estimator():

        #prepare training and validation data
        (x_train,y_train),(x_eval,y_eval) = tf.keras.datasets.mnist.load_data()
        
        # Step 1: Define Pipeline
        pipeline = Pipeline(batch_size=32,
                            feature_name=["x","y"],
                            train_data={"x":x_train,"y":y_train},
                            validation_data={"x":x_eval, "y":y_eval},
                            transform_train=[[Reshape([28,28,1]), Minmax()], 
                                             [Onehot(10)]])

        #Step2: Define Network
        network = Network(model=LeNet(input_name="x",output_name="y"),
                          loss="categorical_crossentropy",
                          metrics=["acc"],
                          optimizer="adam")
                        
        #Step3: Define Estimator
        estimator = Estimator(network=network,
                              pipeline=pipeline,
                              epochs=2)
        return estimator

Pipeline
---------

Overview
^^^^^^^^^

Data pipeline is responsible for providing data from disk/memory to model. It includes
data preprocessing operations before training loop and during training loop. The details of each argument of 
``Pipeline`` can be found in pipeline-api_, but let's talk about several important ones here:

* ``train_data``: The training data in disk (csv) or memory (dict) before creating tfrecords. Ignore it when using existing tfrecords.
* ``validation_data``: The validation data in disk (csv) or memory (dict) before creating tfrecords. It can also be split ratio of ``train_data`` from [0-1]. Ignore it when using existing tfrecords or have no valdiation data.
* ``transform_dataset``: Series of operations on individual features before creating tfrecords.
* ``transform_train``: Series of tensor operations on individual features during training loop.

Preprocess (before training loop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are some preprocessing operations that can be applied before training loop (e.g. Minmax).  
In order to ensure a fast training speed, it is recommended to apply those operations once and for all before training happens. 
FastEstimator offers commonly used preprocessing modules in dynamic-preprocess_ . User can use them in ``transform_dataset`` argument. 

For example, given two features x and y, if you want to proprocess them in the following way before training:

* x: mnist images with size [28,28]--> resize to [50, 50] --> normalize pixel to [0,1]
* y: mnist scalar label, do nothing.

.. code-block:: python
    :emphasize-lines: 10,11

    from fastestimator.pipeline.dynamic.preprocess import Resize, Minmax
    from fastestimator.pipeline.static.preprocess import Reshape, Onehot
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    pipeline = Pipeline(...
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_dataset=[[Resize([50,50]), Minmax()], 
                                           []],
                        ...)

If you have a specific preprocessing needs, you can customize a preprocessing module. 
Let's add some noise to the image before the ``Minmax``:

.. code-block:: python
    :emphasize-lines: 16,17

    from fastestimator.pipeline.dynamic.preprocess import AbstractPreprocessing
    from fastestimator.pipeline.dynamic.preprocess import Resize, Minmax
    import tensorflow as tf
    import numpy as np
    
    class AddNoise(AbstractPreprocessing):
        def transform(self, data, feature=None):
            data = data + 10 * np.random.rand(data.shape[0], data.shape[1])
            return data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    pipeline = Pipeline(...
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_dataset=[[Resize([50,50]), AddNoise(), Minmax()], 
                                           []],
                        ...)

Preprocess (during training loop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the previous example, we added some random noise to the image before training loop. As you may notice,
the drawback is that the random noise not being "random" enough, as we are dealing with image with same noise every iteration.
One natural answer is to add the noise at run time.

Moreover, there are some operations that are better off applied at runtime such as `onehot encoder`. Because we don't want 
our tfrecords to include bunch of useless zeros. Therefore, we need one more type of preprocess that can execute on the fly.

In FastEstimator, on-the-fly preprocessing is implemented in tensorflow, these preprocessing modules are passed through ``transform_train`` argument.
Users can use existing module in static-preprocess_ or customize one for specific needs.

For example, if we want the following process to happen during trainign loop:

* x: reshape the image data from [2500] to [50, 50, 1]
* y: apply one-hot encoder to scalar label

.. code-block:: python
    :emphasize-lines: 12,13

    from fastestimator.pipeline.dynamic.preprocess import Resize, Minmax
    from fastestimator.pipeline.static.preprocess import Reshape, Onehot
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    pipeline = Pipeline(...
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_dataset=[[Resize([50,50]), Minmax()], 
                                           []],
                        transform_train= [[Reshape([50,50,1])], 
                                          [Onehot(10)]])

Next, we add noise on the fly:

.. code-block:: python
    :emphasize-lines: 19,20

    from fastestimator.pipeline.static.preprocess import AbstractPreprocessing
    from fastestimator.pipeline.dynamic.preprocess import Resize, Minmax
    from fastestimator.pipeline.static.preprocess import Reshape, Onehot
    import tensorflow as tf
    import numpy as np

    class AddNoise(AbstractPreprocessing):
        def transform(self, data, decoded_data=None):
            data = data + tf.random.uniform(data.shape)
            return data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    pipeline = Pipeline(...
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_dataset=[[Resize([50,50]), Minmax()], 
                                           []],
                        transform_train= [[Reshape([50,50,1]), AddNoise()], 
                                          [Onehot(10)]])
                        



Augmentation (during training loop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In FastEstimator, augmentation module is same as on-the-fly preprocessing except for one difference:
augmentation allows two or more features to share the same information. For example, in a segmentation task, 
image and mask have to be tranformed in the same manner, meaning that they share the same transformation matrix.

Similar to on-the-fly preprocessing, augmentation is passed through ``transform_train`` argument. FastEstimator provides default 2D-augmentation_,
for example, if we want to apply the following operation to mnist image:

* random rotation between -25 degrees to 25 degrees 
* random zoom between 0.8 and 1.0,

.. code-block:: python
    :emphasize-lines: 7,13

    from fastestimator.pipeline.static.preprocess import Reshape, Onehot
    from fastestimator.pipeline.static.augmentation import Augmentation
    from fastestimator.pipeline.dynamic.preprocess import Minmax
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    aug_obj = Augmentation(rotation_range=25.0, zoom_range=[0.8, 1.0])
    pipeline = Pipeline(...
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_dataset=[[Minmax()], 
                                           []],
                        transform_train= [[Reshape([28,28,1], aug_obj)], 
                                          [Onehot(10)]])

If you want the same augmentation to be applied to other features(when there is mask), just pass the same object:

.. code-block:: python
    :emphasize-lines: 1,4,5

    aug_obj = Augmentation(rotation_range=25.0, zoom_range=[0.8, 1.0])
    pipeline = Pipeline(....
                        feature_name=["image", "mask"],
                        transform_train = [[Reshape([100,100,1]), Minmax(), aug_obj], 
                                          [Reshape([100,100,1]), aug_obj]])


User can also customize augmentation to achieve specific goal, for example, we want to add the 
same random noise for image and mask during training:

.. code-block:: python

    from fastestimator.pipeline.static.augmentation import AbstractAugmentation
    from fastestimator.general.pipeline import Pipeline
    import tensorflow as tf

    # Define customized augmentation
    class AddNoise(AbstractAugmentation):
        def __init__(self, mode="train"):
            self.mode = mode
            self.decoded_data = None
            self.feature_name = None

        def setup(self):
            #we define information shared between features in setup
            self.random_noise = tf.random_uniform([100,100,1], minval=0, maxval=1)

        def transform(self, data):
            data = data + self.random_noise
            return data

    aug_obj = AddNoise()
    pipeline = Pipeline(....
                        feature_name=["image", "mask"],
                        transform_train = [[Reshape([100,100,1]), Minmax(), aug_obj], 
                                          [Reshape([100,100,1]), aug_obj]])

Data Filter
^^^^^^^^^^^^

We can also filter out some example for imbalanced training, FastEstimator provides built-in filter based on scalar feature. For example,
if we want to filter out example with label=1,3,5,7,9:

.. code-block:: python

    from fastestimator.pipeline.static.filter import Filter
    from fastestimator.pipeline.pipeline import Pipeline

    my_filter = Filter(feature_name=["y", "y", "y", "y", "y"],
                       filter_value=[1, 3, 5, 7, 9],
                       keep_prob= [0.0, 0.0, 0.0, 0.0, 0.0])
    pipeline = Pipeline(....
                        data_filter=my_filter)

User can customize their own filter for more complex filters, for example, if we only want to use the example if sum of the image is greater than 10:

.. code-block:: python

    from fastestimator.pipeline.static.filter import Filter
    from fastestimator.pipeline.pipeline import Pipeline

    class my_filter(Filter):
        def __init__(self, mode="train"):
            self.mode = mode

        def predicate_fn(self, dataset):
            #we only use the example when predicate is True
            predicate = tf.greater(tf.reduce_sum(dataset["x"]), 10)
            return predicate
    
    pipeline = Pipeline(....
                        data_filter=my_filter)

Pipeline Debugging
^^^^^^^^^^^^^^^^^^
Once you created your pipeline instace, you can use the `show_batches` method to run the 
pipeline get the outcome in numpy. 

For example, if we want to produce tfrecords first and show two batches of pipeline outcome:

.. code-block:: python

    from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape
    from fastestimator.pipeline.pipeline import Pipeline
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    pipeline = Pipeline(batch_size=32,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        transform_train= [[Reshape([28,28,1]), Minmax()], 
                                          [Onehot(10)]])
    data= pipeline.show_batches(num_batches=2)
    print(data)

If we want to use existing tfrecords and show the outcome of pipleine, we can get rid of 
``train_data`` and provide the tfrecords path in ``show_batches``.

.. code-block:: python

    from fastestimator.pipeline.static.preprocess import Minmax, Onehot, Reshape
    from fastestimator.pipeline.pipeline import Pipeline

    pipeline = Pipeline(batch_size=32,
                        feature_name=["x", "y"],
                        transform_train= [[Reshape([28,28,1]), Minmax()], 
                                          [Onehot(10)]])
    data= pipeline.show_batches(inputs="/your/tfrecord/path", num_batches=2)
    print(data)

Network
-------

Overview
^^^^^^^^^
In FastEstimator, `Network` contains the model architecture and optimization related information. 
User can refer to network-api_ to find detail arguments. `Network` arguments have full compatibility 
with ``tf.keras``, here's one example of network :

.. code-block:: python
    
    from fastestimator.network.network import Network
    from fastestimator.application.lenet import LeNet

    network = Network(model=LeNet(input_name="x",output_name="y"),
                      loss="categorical_crossentropy",
                      metrics=["acc"],
                      optimizer="adam")

Model
^^^^^^

model argument takes an uncompiled ``tf.keras.Model`` instance. In order for pipeline to 
feed data to the correct layer, users have to make sure the Input/Output layer's name matches 
with pipeline's feature name.

.. code-block:: python
    :emphasize-lines: 6,8,12,15

    from fastestimator.pipeline.pipeline import Pipeline
    from fastestimator.network.network import Network
    import tensorflow as tf

    def my_network():
        input_layer = tf.keras.layers.Input(..., name="x")
        .....
        output_layer = tf.keras.layers.Dense(...., name = "y")
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    pipeline = Pipeline(feature_name=["x", "y"],
                        ...)

    network = Network(model=my_network(),
                      ...)
Loss
^^^^^

All standard ``tf.keras`` loss functions are supported, in this case, you can simply provide the name of the loss as a string.
Please refer official `Losses`_ for a complete list of ``tf.keras`` loss functions.

.. code-block:: python

    network = Network(...,
                      loss="categorical_crossentropy",
                      ...)

User can also define customized loss by tensorflow.keras:

.. code-block:: python

    import tensorflow.keras.backend as K

    def rmse_loss(y_true, y_pred):
        rmse = K.sqrt(K.mean(K.square(y_true - y_pred)))
        return rmse

    network = Network(...,
                      loss=rmse_loss,
                      ...)

Metrics
^^^^^^^^

Metrics also work similarly to loss. For standard Keras metrics (e.g., accuracy), you can simply specify the name of metric(s) in a list.
Please refer official `Metrics`_ for a complete list of metric functions.

.. code-block:: python

    network = Network(...,
                      metrics=["acc"],
                      ...)

User can also define customized metrics by tensorflow.keras:

.. code-block:: python

    import tensorflow.keras.backend as K

    def dice(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        coef = (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
        return coef

    network = Network(....
                    metrics=[dice],
                    ...)

Optimizer
^^^^^^^^^^

FastEstimator is compatible with optimizers from ``tf.keras.optimizer``, please refer to Optimizers_
for a compelete list of optimizers. One can simply use string with default optimizer settings:

.. code-block:: python

    network = Network(...,
                      optimizer="adam",
                      ...)

Users can also pass an optimizer instance with custom settings:

.. code-block:: python

    network = Network(...,
                      optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.95),
                      ...)

Model Save Path
^^^^^^^^^^^^^^^
In FastEstimator, model artifacts will be saved in a random path in `/tmp` by default.
If user wants to save model to a specific directory, simply pass the directory to the ``model_dir``:
Once network instance is created, user can access the saving path by ``network.model_dir``.

.. code-block:: python

    network = Network(...,
                      model_dir="/home/model",
                      ...)

Estimator
----------
In FastEstimator, the `Estimator` stores information about the optimization process. It 
takes both `pipeline` and `network` as input, configures them properly for different training 
settings. Next, we show several places where user can customize the training loop.

Callbacks
^^^^^^^^^^
Users can use all callbacks in ``tf.keras.callbacks`` except for three:

* `LearningRateScheduler`: Use ``fastestimator.estimator.callbacks.LearningRateScheduler``
* `ReduceLROnPlateau`: Use ``fastestimator.estimator.callbacks.ReduceLROnPlateau``
* `EarlyStopping`: Use ``fastestimator.estimator.callbacks.EarlyStopping``

For other callbacks, please refer to Callbacks_ in Tensorflow Keras. Next, we present an example of using 
callbacks in FastEstimator:

.. code-block:: python

    from fastestimator.estimator.callbacks import LearningRateScheduler, EarlyStopping
    from fastestimator.network.lrscheduler import CyclicScheduler
    from fastestimator.estimator.estimator import Estimator
    from tensorflow.keras.callbacks import TensorBoard
    
    callbacks = [LearningRateScheduler(schedule=CyclicScheduler()),
                 EarlyStopping(patience=3),
                 TensorBoard()]

    estimator = Estimator(...,
                          callbacks=callbacks,
                          ...)


Custom Steps
^^^^^^^^^^^^
By default, the number of training steps and validation steps for each epoch is calculated as:

* steps_per_epoch = num_examples / batch_size / num_process
* validation_steps = num_examples / batch_size

where `num_process` is the number of parallel training processes, in multi-GPU training, it is the number of GPUs.
User can override the number of training and validation steps in the Estimator:

.. code-block:: python

    estimator = Estimator(...,
                          steps_per_epoch=100,
                          validation_steps=100,
                          ...)

Logging Configuration
^^^^^^^^^^^^^^^^^^^^^
During trainig, the training logs will appear every 100 steps as default, users can change 
the logging interval through ``log_steps`` argument, for example, if we want the log to appear every 10 steps:

.. code-block:: python

    estimator = Estimator(...,
                          log_steps=10,
                          ...)

The training speed may decrease if the logging interval is too small.

Other Utility Functions
------------------------
Outside of core API, FastEstimator offers some useful utilify functions that can be used independently.

TFRecorder
^^^^^^^^^^
``fastestimator.util.tfrecord.TFRecorder`` can help you easily create tfrecord from any data. 
The usage of TFRecorder is very similar to ``Pipeline``:

.. code-block:: python

    from fastestimator.pipeline.dynamic.preprocess import Resize, Minmax
    from fastestimator.util.tfrecord import TFRecorder
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    tfrecorder = TFRecorder(feature_name=["x", "y"],
                            train_data={"x": x_train, "y": y_train},
                            validation_data = 0.2,
                            transform_dataset=[[Resize([50,50]), Minmax()], 
                                                []])
    tfrecorder.create_tfrecord(save_dir="/home/data")

add_summary
^^^^^^^^^^^^
While creating tfrecords, both TFRecorder or FastEstimator produce a summary file that is required for training.
If you have previously built tfrecords outside of FastEstimator or TFRecorder, you can use ``add_summary`` 
to create the summary file for training.

.. code-block:: python

    from fastestimator.util.tfrecord import add_summary, get_features

    # First, get feature name and related information
    print(get_features("/home/data/tf_train_0000.tfrecords"))

    # Next, fill in the required field
    add_summary(data_dir="/data/home/", 
                train_prefix="tf_train_", 
                eval_prefix= "tf_eval_", 
                feature_name=["image_raw", "image_labels"], 
                feature_dtype=["uint8", "int64"])

Full Code Demo
-------------------

* Image Classification: Mnist-Classification_
* Image Segmentation: Cub200-Segmentation_
* Natural Language Processing: IMDB-Review_


.. _pipeline-api: https://github.com/pages/fastestimator/fastestimator/api.html#pipeline
.. _network-api: https://github.com/pages/fastestimator/fastestimator/api.html#network
.. _dynamic-preprocess: https://github.com/pages/fastestimator/fastestimator/api.html#dynamic-preprocess
.. _static-preprocess: https://github.com/pages/fastestimator/fastestimator/api.html#static-preprocess
.. _2D-augmentation: https://github.com/pages/fastestimator/fastestimator/api.html#augmentation
.. _Losses: https://www.tensorflow.org/api_docs/python/tf/keras/losses
.. _Metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
.. _Optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
.. _Callbacks: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
.. _Mnist-Classification: https://github.com/fastestimator/examples/blob/master/classification_mnist/mnist.ipynb
.. _Cub200-Segmentation: https://github.com/fastestimator/examples/blob/master/segmentation_cub200/cub200.ipynb
.. _IMDB-Review: https://github.com/fastestimator/examples/blob/master/sentiment_classification_imdb/imdb.ipynb

