# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Set, Union

from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import parse_modes, to_list, to_set


@traceable()
class Op:
    """A base class for FastEstimator Operators.

    Operators are modular pieces of code which can be used to build complex execution graphs. They are based on three
    main variables: `inputs`, `outputs`, and `mode`. When FastEstimator executes, it holds all of its available data
    behind the scenes in a data dictionary. If an `Op` wants to interact with a piece of data from this dictionary, it
    lists the data's key as one of it's `inputs`. That data will then be passed to the `Op` when the `Op`s forward
    function is invoked (see NumpyOp and TensorOp for more information about the forward function). If an `Op` wants to
    write data into the data dictionary, it can return values from its forward function. These values are then written
    into the data dictionary under the keys specified by the `Op`s `outputs`. An `Op` will only be run if its associated
    `mode` matches the current execution mode. For example, if an `Op` has a mode of 'eval' but FastEstimator is
    currently running in the 'train' mode, then the `Op`s forward function will not be called.

    Normally, if a single string "key" is passed as `inputs` then the value that is passed to the forward function will
    be the value exactly as it is stored in the data dictionary: dict["key"]. On the other hand, if ["key"] is passed as
    `inputs` then the value passed to the forward function will be the element stored in the data dictionary, but
    wrapped within a list: [dict["key"]]. This can be inconvenient in some cases where an `Op` is anticipated to take
    one or more inputs and treat them all in the same way. In such cases the `in_list` member variable may be manually
    overridden to True. This will cause data to always be sent to the forward function like [dict["key"]] regardless of
    whether `inputs` was a single string or a list of strings. For an example of when this is useful, see:
    fe.op.numpyop.univariate.univariate.ImageOnlyAlbumentation.

    Similarly, by default, if an `Op` has a single `output` string "key" then that output R will be written into the
    data dictionary exactly as it is presented: dict["key"] = R. If, however, ["key"] is given as `outputs` then the
    return value for R from the `Op` is expected to be a list [X], where the inner value will be written to the data
    dictionary: dict["key"] = X. This can be inconvenient in some cases where an `Op` wants to always return data in a
    list format without worrying about whether it had one input or more than one input. In such cases the `out_list`
    member variable may be manually overridden to True. This will cause the system to always assume that the response is
    in list format and unwrap the values before storing them into the data dictionary. For an example, see:
    fe.op.numpyop.univariate.univariate.ImageOnlyAlbumentation.

    Args:
        inputs: Key(s) from which to retrieve data from the data dictionary.
        outputs: Key(s) under which to write the outputs of this Op back to the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    inputs: List[Union[str, Callable]]
    outputs: List[str]
    mode: Set[str]
    in_list: bool  # Whether inputs should be presented as a list or an individual value
    out_list: bool  # Whether outputs will be returned as a list or an individual value

    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None) -> None:
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = parse_modes(to_set(mode))
        self.in_list = not isinstance(inputs, (str, Callable))
        self.out_list = not isinstance(outputs, str)


def get_inputs_by_op(op: Op, store: Mapping[str, Any]) -> Any:
    """Retrieve the necessary input data from the data dictionary in order to run an `op`.

    Args:
        op: The op to run.
        store: The system's data dictionary to draw inputs out of.

    Returns:
        Input data to be fed to the `op` forward function.
    """
    data = None
    if op.inputs:
        data = [store[key] if not isinstance(key, Callable) else key() for key in op.inputs]
        if not op.in_list:
            data = data[0]
    return data


def write_outputs_by_op(op: Op, store: MutableMapping[str, Any], outputs: Any) -> None:
    """Write `outputs` from an `op` forward function into the data dictionary.

    Args:
        op: The Op which generated `outputs`.
        store: The data dictionary into which to write the `outputs`.
        outputs: The value(s) generated by the `op`s forward function.
    """
    if not op.out_list:
        outputs = [outputs]
    for key, data in zip(op.outputs, outputs):
        store[key] = data
