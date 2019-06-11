FastEstimator API reference
============================

Pipeline
---------

.. autoclass:: fastestimator.pipeline.pipeline.Pipeline
    :members:
    :undoc-members:
    :show-inheritance:

Dynamic Preprocess
-------------------

AbstractPreprocessing
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing
    :members:
    :undoc-members:
    :show-inheritance:

NrrdReader
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.NrrdReader
    :members:
    :undoc-members:
    :show-inheritance:

ImageReader
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.ImageReader
    :members:
    :undoc-members:
    :show-inheritance:

Zscore
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Zscore
    :members:
    :undoc-members:
    :show-inheritance:

Minmax
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Minmax
    :members:
    :undoc-members:
    :show-inheritance:

Scale
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Scale
    :members:
    :undoc-members:
    :show-inheritance:

Onehot
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Onehot
    :members:
    :undoc-members:
    :show-inheritance:

Resize
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Resize
    :members:
    :undoc-members:
    :show-inheritance:

Reshape
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.dynamic.preprocess.Reshape
    :members:
    :undoc-members:
    :show-inheritance:

Static Preprocess
-------------------

AbstractPreprocessing
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.AbstractPreprocessing
    :members:
    :undoc-members:
    :show-inheritance:

Binarize
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Binarize
    :members:
    :undoc-members:
    :show-inheritance:

Zscore
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Zscore
    :members:
    :undoc-members:
    :show-inheritance:

Minmax
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Minmax
    :members:
    :undoc-members:
    :show-inheritance:

Scale
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Scale
    :members:
    :undoc-members:
    :show-inheritance:

Onehot
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Onehot
    :members:
    :undoc-members:
    :show-inheritance:

Resize
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Resize
    :members:
    :undoc-members:
    :show-inheritance:

Reshape
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.preprocess.Reshape
    :members:
    :undoc-members:
    :show-inheritance:

Augmentation
-------------------

AbstractAugmentation
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.augmentation.AbstractAugmentation
    :members:
    :undoc-members:
    :show-inheritance:

Augmentation
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.pipeline.static.augmentation.Augmentation
    :members:
    :undoc-members:
    :show-inheritance:

Filter
-------------------

.. autoclass:: fastestimator.pipeline.static.filter.Filter
    :members:
    :undoc-members:
    :show-inheritance:

Cyclic Learning Rate
----------------------

.. autoclass:: fastestimator.network.lrscheduler.CyclicScheduler
    :members:
    :undoc-members:
    :show-inheritance:

Network
----------------------

.. autoclass:: fastestimator.network.network.Network
    :members:
    :undoc-members:
    :show-inheritance:

Estimator
----------------------

.. autoclass:: fastestimator.estimator.estimator.Estimator
    :members:
    :undoc-members:
    :show-inheritance:

Callbacks
----------------------

OutputLogger
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.estimator.callbacks.OutputLogger
    :members:
    :undoc-members:
    :show-inheritance:

LearningRateUpdater
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.estimator.callbacks.LearningRateUpdater
    :members:
    :undoc-members:
    :show-inheritance:

LearningRateScheduler
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.estimator.callbacks.LearningRateScheduler
    :members:
    :undoc-members:
    :show-inheritance:

ReduceLROnPlateau
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.estimator.callbacks.ReduceLROnPlateau
    :members:
    :undoc-members:
    :show-inheritance:

TFRecord Utility Functions
----------------------------

TFRecorder
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.util.tfrecord.TFRecorder
    :members:
    :undoc-members:
    :show-inheritance:

tfrecord_to_np
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.util.tfrecord.tfrecord_to_np
    :members:
    :undoc-members:
    :show-inheritance:

get_number_of_examples
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.util.tfrecord.get_number_of_examples
    :members:
    :undoc-members:
    :show-inheritance:

get_features
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.util.tfrecord.get_features
    :members:
    :undoc-members:
    :show-inheritance:

add_summary
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastestimator.util.tfrecord.add_summary
    :members:
    :undoc-members:
    :show-inheritance: