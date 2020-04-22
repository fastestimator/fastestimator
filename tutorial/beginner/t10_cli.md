# Tutorial 10: How to use FastEstimator CLI

## Overview
FastEstimator comes with a set of CLI tools that can help users train and test their models quickly. The FastEstimator CLI currently support the following commands:

* train
* test
* logs

In this tutorial, we will go over each of those and the arguments they take.

## fastestimator train
We use the train command in the CLI to train the ML models. The following snippet shows how the command can be used:

`fastestimator train [arguments] <entry_point>`

The entry point here repersents the path to the python file that contains the estimator definition, using the `get_estimator()` function. The optional arguments for this command are the following:
* hyperparameters: This is an optional argument which takes path to a json file containing all the training hyperparameters like epochs, batch_size, optimizer, etc. This is useful when you want to repeat the training more than once and the list of the hyperparameter is getting really long.
* epochs: Instead of using a json configuration file you can also optionally pass in epochs using this argument.
* batch_size: Optional batch size for training.
* optimizer: Another optional argument that specifies which optimizer to use for the training.

## fastestimator test
The test command is similar to the train command with the difference that it's used when we just want to test a model instead of train it. The following snippet shows how the command can be used:

`fastestimator test [options] <entry_point>`

All the details for the command are the same as the train command.

## fastestimator logs
This command is used to generate comparison graphs amongst one or more log file. The following shows the syntax for this command:

`fastestimator logs [options] <log file(s) directory>`

The log file directory is the path to a folder containing one or more log files. The optional arguments for this command are the following:
* extension: The file type / extension of your logs. Defaults to `.txt`.
* recursive: Recursively search sub-directories for log files.
* ignore: The names of metrics to ignore though they may be present in the log files.
* smooth: The amount of gaussian smoothing to apply (zero for no smoothing). Defaults to `1`.
* pretty_names: Clean up the metric names for display.
* common_legend: Generate one legend total. Defaults to `True` (belongs to `legend arguments` group).
* split_legend: Generate one legend per graph. Defaults to `False` (belongs to `legend arguments` group).
* save: Save the output image. May be accompanied by a directory into which the file is saved. If no output directory is specified, the log directory will be used. Defaults to `False` (belongs to `output arguments` group).
* display: Render the image to the UI rather than saving it. Defaults to `True` (belongs to `output arguments` group).