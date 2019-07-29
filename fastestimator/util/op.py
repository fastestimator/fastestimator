from fastestimator.record.preprocess import NumpyPreprocess
from fastestimator.pipeline.preprocess import TensorPreprocess
from fastestimator.pipeline.augmentation import TensorAugmentation
from fastestimator.pipeline.filter import TensorFilter
from fastestimator.network.network import Model
from itertools import chain

def flatten_operation(ops):
    if not isinstance(ops, list):
        ops = [ops]
    for idx, op in enumerate(ops):
        if not isinstance(op, list):
            ops[idx] = [op]
    ops = list(chain.from_iterable(ops))
    return ops

def get_op_from_mode(ops, current_mode):
    selected_ops = []
    for op in ops:
        op_mode = op.mode
        if not isinstance(op_mode, list):
            op_mode = [op_mode]
        if None in op_mode or current_mode in op_mode:
            selected_ops.append(op)
    return selected_ops

def verify_ops(ops, class_name):
    inheritage = {"RecordWriter": NumpyPreprocess,
                   "Pipeline": (TensorPreprocess, TensorAugmentation, TensorFilter),
                   "Network": Model}
    inheritage_class = inheritage[class_name]
    assert ops[0].inputs, "must provide inputs for the operation '{}' in '{}'".format(type(ops[0]).__name__, class_name)
    assert ops[-1].outputs, "must provide outputs for the operation '{}' in '{}'".format(type(ops[-1]).__name__, class_name)
    inputs = ops[0].inputs
    for idx, op in enumerate(ops):
        assert isinstance(op, inheritage_class), "operation '{}' in class '{}' doesn't have correct inheritage".format(type(op).__name__, class_name)
        if idx +1 < len(ops) and ops[idx+1].inputs:
            old_inputs = op.inputs
            new_inputs = ops[idx+1].inputs
            if new_inputs and new_inputs != inputs:
                assert op.outputs, "must provide outputs for the operation '{}' in class '{}', otherwise the result will be lost".format(type(op).__name__, class_name)