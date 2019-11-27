# Learn about FastEstimator

**Welcome to FastEstimator Tutorials!**

In this section, we will guide you through different functionnalities of FastEstimator to help you get started, as well as leverage advanced features such as Scheduler or Interpretation. 
But before jumping to the first tutorial, we want to highlight what differentiates FastEstimator from other frameworks.
There are three main concepts behind FastEstimator: the structure of model development around 3 APIs, the concept of Operator and the concept of Trace.

## Three  main APIs to build a model
All deep learning training workﬂows involve three essential components: data pipeline, network, and optimization strategy. Data pipeline extracts data from disk to RAM, performs transformations, and then loads the data onto the device. Network stores trainable and differentiable graphs. Optimization strategy combines data pipeline and network in an iterative process. Each of these components represents a critical API in FastEstimator: Pipeline, Network, and Estimator. Users will interact with these three APIs for any deep learning task. 

* *Pipeline*:   
Pipeline can be summarized as an Extraction-Transformation-Load (ETL) process. The extractor can take data either from disk or RAM, with features being either paired or unpaired (e.g., domain adaptation). The transformer builds graphs for preprocessing. The data utility provides support for scenarios like imbalanced training, feature padding, distributed training, and progressive training.

* *Network*:  
Network manages trainable models. First,the constructor builds model graphs and creates timestamps on these graphs in the case of progressive training. The transformer then connects different pieces of model graphs and non-trainable graphs together. The updater tracks and applies gradients to each trainable model.

* *Estimator*:  
Estimator is responsible for the training loop. Before training starts, a smoke test is performed on all graphs to detect potential run-time errors as well as to warm up the graph for faster execution. It then proceeds with training, generating any user-speciﬁed output along the way. 

<img width="800" alt="Capture" src="https://user-images.githubusercontent.com/46055963/69750487-bf479380-1101-11ea-9b33-fd7937c1ba75.PNG">

The central component of both Pipeline and Network is a sequence of Operators. Similarly for the Estimator, there is a sequence of Traces. Operator and Trace are both core concepts which differentiate FastEstimator from other frameworks.

## Operator: the highest level of flexibility

The common goal of all high-level deep learning APIs is to enable complex graph building with less code. For that, most frameworks introduce the concept of layers (aka blocks and modules) to simplify network deﬁnition. However, as model complexity increases, even layer representations may become undesirably verbose.
Therefore, we propose the concept of Operator, a higher level abstraction for layers, to achieve code reduction without losing ﬂexibility. An Operator represents a task-level computation module for data (in the form of key:value pairs), which can be either trainable or non-trainable. Every Operator has three components: input key(s), transformation function, and output key(s). 
The execution ﬂow of a single Operator involves: 
1) take the value of the input key(s) from the batch data   
2) apply transformation functions to the input value   
3) write the output value back to the batch data with output key(s).    
With the help of Operators, complex computational graphs can be built using only a few lines of code.

Below is an example of a deep learning application expressed as a sequence of Operators:
<img width="800" alt="op" src="https://user-images.githubusercontent.com/46055963/69750600-f6b64000-1101-11ea-925c-7b1fd63fa619.PNG">

## Trace: a training loop controller
 In FastEstimator, both metrics and callbacks are uniﬁed into Traces, our training loop controller. Metrics are quantitative measures of model performance and are computed during the training or validation loop. Callbacks are modules that contain event functions like on_epoch_begin and on_batch_begin, which allow users to insert custom functions to be executed at different locations within the training loop. Implementation-wise, since metrics and callbacks are usually separate, callbacks in most frameworks are not designed to have easy access to batch data. As a result, researchers may have to use less efﬁcient workarounds to access intermediate results produced within the training loop. Moreover, callbacks are not designed to communicate with each other, which adds further inconvenience if a later callback needs the outputs from a previous callback.
Trace in FastEstimator preserves the event functions in callbacks and provides the following improvements: 
1) Traces have easy access to batch data directly from the batch loop
2) Every Trace can pass data to later Traces to increase re-usability of results
3) Metric computation can leverage batch data directly without a graph
4) Metrics can be accumulated through Trace member variables without update rules. 
These improvements brought by Trace have enabled many new functionalities. For example, our model interpretation module is made possible by easy batch data access. Furthermore, Trace has access to all API components such that changing model architecture or data pipeline within the training loop is straightforward. 
 
![trace](https://user-images.githubusercontent.com/46055963/69750660-249b8480-1102-11ea-84b5-8f1523136cc3.png)
 
  
 Already comfortable with FastEstimator? Check our state-of-the-art implementations!
