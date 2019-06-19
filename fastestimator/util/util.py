import argparse
import os
import re
import string
import subprocess

import tensorflow as tf


def get_num_GPU():
    """
    Gets number of GPUs on device

    Returns:
        Number of GPUS available on device
    """
    try:
        result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = [line.split() for line in result.splitlines() if line.startswith("Attached GPUs")]
        num_gpu = int(lines[0][-1])
    except:
        num_gpu = 0
    return num_gpu


def convert_tf_dtype(datatype):
    """
    Gets the tensorflow datatype from string
    
    Args:
        datatype: String of datatype

    Returns:
        Tensor data type
    """
    datatype_map = {"string": tf.string,
                    "int8": tf.int8,
                    "uint8": tf.uint8,
                    "int16": tf.int16,
                    "uint16": tf.uint16,
                    "int32": tf.int32,
                    "uint32": tf.uint32,
                    "int64": tf.int64,
                    "uint64": tf.uint64,
                    "float16": tf.float16,
                    "float32": tf.float32,
                    "float64": tf.float64}
    return datatype_map[datatype]


def strip_suffix(target, suffix):
    """
    Remove the given suffix from the target if it is present there

    Args:
        target: A string to be formatted
        suffix: A string to be removed from 'target'

    Returns:
        The formatted version of 'target'
    """
    if suffix is None or target is None:
        return target
    s_len = len(suffix)
    if target[-s_len:] == suffix:
        return target[:-s_len]
    return target


def prettify_metric_name(metric):
    """
    Add spaces to camel case words, then swap _ for space, and capitalize each word

    Args:
        metric: A string to be formatted

    Returns:
        The formatted version of 'metric'
    """
    return string.capwords(re.sub("([a-z])([A-Z])", r"\g<1> \g<2>", metric).replace("_", " "))


def remove_blacklist_keys(dic, blacklist):
    """
    A function which removes the blacklisted elements from a dictionary

    Args:
        dic: The dictionary to inspect
        blacklist: keys to be removed from dic if they are present

    Returns:
        None

    Side Effects:
        Entries in dic may be removed
    """
    if blacklist is None:
        return
    for elem in blacklist:
        dic.pop(elem, None)


def is_number(s):
    """
    Args:
        s: A string
    Returns:
        True iff the string represents a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


class SaveAction(argparse.Action):
    """
    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
        is invoked directly during argument parsing.
    """

    def __init__(self, option_strings, dest, nargs='?', **kwargs):
        if '?' != nargs:
            raise ValueError("nargs must be \'?\'")
        super(SaveAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, self.dest + '_dir', values if values is None else os.path.join(values, ''))
