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
import dis
import inspect
import re
import types
from collections import deque, namedtuple
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.util.util import strip_prefix

_Function = namedtuple('_Function', ['func', 'name'])
_BoundFn = namedtuple('_BoundFn', ['func', 'args'])
_PartialBind = namedtuple('_PartialBind', ['args', 'kwargs'])
_Command = namedtuple('_Command', ['left', 'right', 'command'])
_Condition = namedtuple('_Condition', ['left', 'right', 'condition'])
_VarWrap = namedtuple('_VarWrap', ['var'])
_ChunkSpec = namedtuple('_ChunkSpec', ['chunk_start', 'idx_start', 'chunk_mid', 'idx_mid', 'chunk_end', 'idx_end'])
_CommandTable = {
    'POWER': '**',
    'MULTIPLY': '*',
    'MATRIX_MULTIPLY': '@',
    'FLOOR_DIVIDE': '//',
    'TRUE_DIVIDE': '/',
    'MODULO': '%',
    'ADD': '+',
    'SUBTRACT': '-',
    'SUBSCR': '[]',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'AND': '&',
    'XOR': '^',
    'OR': '|',
    '>': '>',
    '<': '<',
    '==': '==',
    '!=': '!=',
    '<=': '<=',
    '>=': '>='
}

Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


def _deref_is_callable(instruction: dis.Instruction, closure_vars: inspect.ClosureVars) -> bool:
    """A function to determine whether an `instruction` is referencing something that is callable or not.

    Args:
        instruction: The instruction to be investigated.
        closure_vars: The variables in the current scope.

    Returns:
        True iff the `instruction` is pointing to a callable object.
    """
    deref = closure_vars.nonlocals.get(
        instruction.argval,
        closure_vars.globals.get(instruction.argval, closure_vars.builtins.get(instruction.argval, None)))
    return hasattr(deref, '__call__')


def _trace_value(inp: Any, wrap_str: bool = True, include_id: bool = True) -> str:
    """Convert an input value to it's string representation.

    Args:
        inp: The input value to be converted.
        wrap_str: Whether literal string values should be wrapped inside extra quote marks.
        include_id: Whether to include object ids when representing objects.

    Returns:
        A string representation of the input.
    """
    if isinstance(inp, str):
        return f"'{inp}'" if wrap_str else inp
    elif hasattr(inp, '_fe_traceability_summary'):
        # The first time a traceable object goes through here it won't have it's summary instantiated yet, so it will
        # fall through to the class check at the end to get it's id.
        # noinspection PyProtectedMember
        return inp._fe_traceability_summary
    elif isinstance(inp, (int, float, bool, type(None))):
        return f"{inp}"
    elif inspect.ismethod(inp):
        return f"the '{inp.__name__}' method of ({_trace_value(inp.__self__, wrap_str, include_id)})"
    elif inspect.isfunction(inp) or inspect.isclass(inp):
        if inspect.isfunction(inp) and inp.__name__ == "<lambda>":
            code = inp.__code__
            var_names = code.co_varnames
            # Attempt to figure out what the lambda function is doing. If it is being used only to invoke some other
            # function (like one might do with LRScheduler), then the parse should work.
            func_description = _parse_lambda(inp)
            if func_description is None:
                return "a lambda function passing {}".format(
                    str.join(", ", map(lambda x: "'{}'".format(x), var_names)) or 'no arguments')
            else:
                return "a lambda function passing {} to: {}".format(
                    str.join(", ", map(lambda x: "'{}'".format(x), var_names)) or 'no arguments',
                    _trace_value(func_description,
                                 wrap_str=False if isinstance(func_description, str) else True,
                                 include_id=include_id))
        else:
            return f"{inp.__module__}.{inp.__qualname__}"
    elif isinstance(inp, _Function):
        if inspect.isbuiltin(inp.func) or not hasattr(inp.func, '__module__') or not hasattr(inp.func, '__qualname__'):
            return inp.name
        else:
            return f"{inp.func.__module__}.{inp.func.__qualname__}"
    elif isinstance(inp, _PartialBind):
        s1, join, s2 = "", "", ""
        if inp.args:
            s1 = "args={}".format(_trace_value(inp.args, wrap_str=True, include_id=include_id))
        if inp.kwargs:
            join = ", kwargs=" if s1 else ""
            s2 = "{}".format(_trace_value(inp.args, wrap_str, include_id))
        return f"{s1}{join}{s2}"
    elif isinstance(inp, _Command):
        return "{} {} {}".format(_trace_value(inp.left, wrap_str, include_id),
                                 inp.command,
                                 _trace_value(inp.right, wrap_str, include_id))
    elif isinstance(inp, _Condition):
        return "{} if {} else {}".format(_trace_value(inp.left, wrap_str, include_id),
                                         _trace_value(inp.condition, wrap_str, include_id),
                                         _trace_value(inp.right, wrap_str, include_id))
    elif isinstance(inp, _BoundFn):
        return "{} invoked with: {}".format(_trace_value(inp.func, wrap_str, include_id=False),
                                            _trace_value(inp.args, wrap_str=False, include_id=include_id))
    elif isinstance(inp, inspect.BoundArguments):
        return _trace_value(inp.arguments, wrap_str=False, include_id=include_id)
    elif isinstance(inp, _VarWrap):
        return inp.var
    elif isinstance(inp, (tf.keras.Model, torch.nn.Module)):
        # FE models should never actually get here since they are given summaries by trace_model() during fe.build()
        name = inp.model_name if hasattr(inp, 'model_name') else "<Unknown Model Name>"
        id_str = f" (id: {id(inp)})" if include_id else ""
        return f"a neural network model named '{name}'{id_str}"
    # The collections need to go after _Command and _BoundFn
    elif isinstance(inp, List):
        return "[{}]".format(", ".join([_trace_value(x, wrap_str, include_id) for x in inp]))
    elif isinstance(inp, Tuple):
        return "({})".format(", ".join([_trace_value(x, wrap_str, include_id) for x in inp]))
    elif isinstance(inp, Set):
        return "{{{}}}".format(", ".join([_trace_value(x, wrap_str, include_id) for x in inp]))
    elif isinstance(inp, Mapping):
        return "{{{}}}".format(", ".join([
            "{}={}".format(_trace_value(k, wrap_str=wrap_str, include_id=include_id),
                           _trace_value(v, wrap_str=True, include_id=True)) for k,
            v in inp.items()
        ]))
    elif isinstance(inp, (tf.Tensor, torch.Tensor, np.ndarray, tf.Variable)):
        id_str = f" (id: {id(inp)})" if include_id or isinstance(inp, tf.Variable) else ""
        if isinstance(inp, (tf.Tensor, torch.Tensor, tf.Variable)):
            inp = inp.numpy()
        rank = inp.ndim
        if rank > 1 or (rank == 1 and inp.shape[0] > 5):
            return f"an array with shape {inp.shape}{id_str}"
        return f"{inp}{id_str}"
    # This should be the last elif
    elif hasattr(inp, '__class__'):
        inp = inp.__class__
        id_str = f" (id: {id(inp)})" if include_id else ""
        return f"an instance of {inp.__module__}.{inp.__qualname__}{id_str}"
    else:
        id_str = f" with id: {id(inp)})" if include_id else ""
        return f"an object{id_str}"


def _traverse_chunks(lambda_specs: List[_ChunkSpec],
                     chunks: List[str],
                     chunk_start: int,
                     idx_start: int,
                     end_char: str,
                     closure_exit: bool = False,
                     include_last: bool = True) -> Tuple[Optional[int], Optional[int]]:
    """Move through `chunks` looking for the end of a lambda function component.

    Args:
        lambda_specs: Existing known lambda functions (to be skipped over during chunk parsing).
        chunks: A string representation of a lambda function which has been broken into chunks, where each chunk
            contains either code or a string literal, but not both.
        chunk_start: The chunk idx to begin the search.
        idx_start: The idx within a chunk in which to begin the search.
        end_char: The character which is used to mark the end of the search.
        closure_exit: Whether the search can end before finding the `end_char` in the event that a bracket imbalance is
            detected.
        include_last: Whether to include the last character (usually `end_char`) in the returned indices.

    Returns:
        A tuple containing two elements: 1) The chunk idx of the end of the component. 2) The idx of the end of the
        component within the chunk. These values will both be None if the end cannot be found.
    """
    chunk_end, idx_end = None, None
    open_char = 0
    chunk_idx = chunk_start
    idx = idx_start
    while chunk_end is None:
        if chunk_idx >= len(chunks):
            return None, None  # Could not find the end
        chunk = chunks[chunk_idx]
        if chunk.startswith("'") or chunk.startswith('"'):
            # Skip over string chunks
            chunk_idx += 1
            idx = 0
            continue
        while idx < len(chunk):
            for spec in lambda_specs:
                if spec.chunk_start == chunk_idx and idx == spec.idx_start:
                    # We ran into a different lambda function, so skip over it
                    chunk_idx = spec.chunk_end
                    chunk = chunks[chunk_idx]
                    idx = spec.idx_end
            char = chunk[idx]
            if char in ('(', '{', '['):
                open_char += 1
            elif char in (')', '}', ']'):
                open_char -= 1
                if open_char < 0:
                    if closure_exit:
                        return chunk_idx, idx
                    return None, None
            elif char == end_char and open_char == 0:
                chunk_end = chunk_idx
                idx_end = idx + include_last
                break
            idx += 1
        chunk_idx += 1
        idx = 0
    return chunk_end, idx_end


def _combine_chunks(chunks: List[str], specs: List[_ChunkSpec],
                    skip: str = "ambda") -> List[Tuple[Set[str], Set[str], str]]:
    """Recombine a series of `chunks` into individual lambda expressions based on the given `specs`.

    Args:
        chunks: The chunks to be recombined.
        specs: The indices by which to recombine the chunks to get the desired lambda expression.
        skip: A prefix to be cut off when examining the re-combined lambda expression to extract its input args.

    Returns:
        A list of tuples, one per `spec`. Each tuple contains three elements: 1) The names of the variables used by the
        lambda expression. 2) The string constants used in the lambda expression. 3) The string representation of the
        lambda function implementation.
    """
    results = []
    for spec in specs:
        strings = set()
        # Put together the first piece
        if spec.chunk_start == spec.chunk_mid:
            part1 = chunks[spec.chunk_start][spec.idx_start:spec.idx_mid]
        else:
            part1 = chunks[spec.chunk_start][spec.idx_start:]
            for i in range(spec.chunk_start + 1, spec.chunk_mid):
                if chunks[i].startswith("'") or chunks[i].startswith('"'):
                    strings.add(chunks[i][1:-1])  # Take off the outer quote marks
                    continue
                part1 = part1 + chunks[i]
            part1 = part1 + chunks[spec.chunk_mid][:spec.idx_mid]
        if part1.find(skip) != -1:
            part1 = part1[part1.find(skip) + len(skip):]
        part1 = part1.strip()
        part1 = _extract_args(part1)
        # Put together the second piece
        if spec.chunk_mid == spec.chunk_end:
            part2 = chunks[spec.chunk_mid][spec.idx_mid:spec.idx_end]
        else:
            part2 = chunks[spec.chunk_mid][spec.idx_mid:]
            for i in range(spec.chunk_mid + 1, spec.chunk_end):
                part2 = part2 + chunks[i]
                if chunks[i].startswith("'") or chunks[i].startswith('"'):
                    strings.add(chunks[i][1:-1])  # Take off the outer quote marks
            part2 = part2 + chunks[spec.chunk_end][:spec.idx_end]
        part2 = part2.strip()
        results.append((part1, strings, part2))
    return results


def _extract_args(input_str: str) -> Set[str]:
    """Extract the argument names from a string representation of a lambda function.

    Args:
        input_str: The string to be inspected. Something like 'x, opt=5:".

    Returns:
        The argument names from the `input_str`. Something like {'x', 'opt'}.
    """
    results = set()
    arg = ''
    open_char = 0
    left_side = True
    for char in input_str:
        if open_char == 0 and char == ":":
            arg = arg.strip()
            if arg:
                results.add(arg)
            break
        if char in ('(', '{', '['):
            open_char += 1
        elif char in (')', '}', ']'):
            open_char -= 1
        elif char == '=' and open_char == 0:
            arg = arg.strip()
            if arg:
                results.add(arg)
            arg = ''
            left_side = False
        elif char == ',' and open_char == 0:
            left_side = True
        elif left_side:
            arg += char
    return results


def _parse_lambda_fallback(function: types.FunctionType) -> Optional[str]:
    """Convert a lambda function into a string representation, disambiguating variables when possible.

    Args:
        function: The lambda function to be inspected.

    Returns:
        A string representation of the lambda function, along with information about the variables it references when
        possible. If this fails (the lambda function was defined in a REPL environment, or more than one lambda function
        was defined in the same line of code and the results cannot be disambiguated) then None will be returned.
    """
    try:
        source = inspect.getsource(function)
    except OSError:
        return None
    # Remove trailing space
    source = source.strip()
    # Find string literal regions in order to skip over them during parsing
    regions = []
    open_char = ''
    open_idx = 0
    for idx, char in enumerate(source):
        if char in ("'", '"') and (idx == 0 or source[idx - 1] != '\\'):
            # We have a non-escaped quote mark
            if not open_char:
                # We are opening a quote
                regions.append((open_idx, idx))
                open_idx = idx
                open_char = char
            elif open_char == char:
                # We are closing an open string
                regions.append((open_idx, idx + 1))
                open_idx = idx + 1
                open_char = ''
    if open_idx < len(source):
        regions.append((open_idx, len(source)))
    chunks = [source[r[0]:r[1]] for r in regions]
    # Do some preliminary cleaning to help with multi-line lambdas
    for idx, chunk in enumerate(chunks):
        if chunk.startswith("'") or chunk.startswith('"'):
            # Ignore string chunks
            continue
        # Clean up line breaks and extra spaces
        chunk = chunk.replace('\n', '')
        chunk = chunk.replace('\r', '')
        chunk = chunk.replace('\\', '')
        chunk = re.sub('\t', ' ', chunk)
        chunk = re.sub('[ ]+', ' ', chunk)
        chunks[idx] = chunk
    # Find the lambda functions
    lambda_idx = []
    for chunk_idx, chunk in enumerate(chunks):
        if chunk.startswith("'") or chunk.startswith('"'):
            # Ignore string chunks
            continue
        lambda_idx.extend([(chunk_idx, m.start() + 1) for m in re.finditer(r'(^|[\W])lambda[ :]', chunk)])
    # Figure out the extents of the lambda functions, starting from the back
    lambda_specs = []
    for fn in reversed(lambda_idx):
        chunk_start, idx_start = fn
        # Crawl over the arguments to find where the lambda definition starts
        chunk_def_start, idx_def_start = _traverse_chunks(lambda_specs, chunks, chunk_start, idx_start, end_char=':')
        if chunk_def_start is None:
            return None  # The lambda function syntax is invalid, allow python to crash later
        # Crawl over the function to find where the lambda definition ends
        chunk_def_end, idx_def_end = _traverse_chunks(lambda_specs, chunks, chunk_def_start, idx_def_start,
                                                      end_char=',', closure_exit=True, include_last=False)
        if chunk_def_end is None:
            return None  # The lambda function syntax is invalid, allow python to crash later
        lambda_specs.insert(
            0, _ChunkSpec(chunk_start, idx_start, chunk_def_start, idx_def_start, chunk_def_end, idx_def_end))
    # Extract the lambda functions
    lambda_fns = _combine_chunks(chunks, lambda_specs)
    code = function.__code__
    lambda_fn = None
    # Figure out which lambda function corresponds to the user's lambda function
    if len(lambda_fns) == 1:
        lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Throw out any functions which don't have the right var names
        args = set(code.co_varnames)
        lambda_fns = list(filter(lambda x: x[0] == args, lambda_fns))
        if len(lambda_fns) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Throw out any functions which don't have necessary strings
        strings = set(filter(lambda x: isinstance(x, str) and '<lambda>.<locals>' not in x, code.co_consts))
        lambda_fns = list(filter(lambda x: strings.issubset(x[1]), lambda_fns))
        if len(lambda_fns) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Throw out any functions which don't contain the right references
        names = set(code.co_names)
        lambda_fns = list(filter(lambda x: all(s in x[2] for s in names), lambda_fns))
        if len(lambda_fns) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Maybe it's the same lambda function multiple times...
        if len({fn[2] for fn in lambda_fns}) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Couldn't figure out which lambda function this is
        return None
    # De-reference any variables
    refs = code.co_freevars
    lam = lambda_fn[2]
    if refs:
        closure_vars = inspect.getclosurevars(function)
        ref_map = {
            ref:
            closure_vars.nonlocals.get(ref,
                                       closure_vars.globals.get(ref, closure_vars.builtins.get(ref, _VarWrap(ref))))
            for ref in refs
        }
        lam = f"{lam} where {_trace_value(ref_map, wrap_str=False)}"
    return lam


def _parse_lambda(function: types.FunctionType) -> Optional[Any]:
    """Convert a lambda function into its argument-based representation.

    The `function` is expected to be a lambda expression, which means that the set of bytecode instructions is limited
    compared to examining any possible function.

    Args:
        function: A lambda function to be inspected

    Returns:
        The arguments being used to invoke `function`, or None if parsing fails.
    """
    code = function.__code__
    instructions = [x for x in dis.get_instructions(code)]
    closure_vars = inspect.getclosurevars(function)  # The variables defining the current scope.
    conditions = []
    args = []
    idx = 0
    while idx < len(instructions):
        instruction = instructions[idx]
        if instruction.opname == 'RETURN_VALUE':
            # Lambda functions don't support the return keyword, instead values are returned implicitly
            if conditions:
                current_condition = conditions.pop()
                arg = args.pop()
                instructions.pop(idx - 1)
                idx -= 1
                # In lambda functions, conditions always fill in the order: condition -> left -> right
                if current_condition.left is None:
                    conditions.append(_Condition(left=arg, condition=current_condition.condition, right=None))
                    instructions.pop(idx - 1)
                    idx -= 1
                else:
                    args.append(
                        _Condition(left=current_condition.left, condition=current_condition.condition, right=arg))
                    if conditions:
                        # The return value can be used to satisfy a condition slot
                        idx -= 1
            else:
                break
        elif instruction.opname == 'LOAD_CONST':
            # It's a constant value
            args.append(instruction.argval)
        elif instruction.opname == 'LOAD_FAST':
            # It's a variable from a lambda expression
            args.append(_VarWrap(instruction.argval))
        elif instruction.opname in ('BUILD_LIST', 'BUILD_TUPLE', 'BUILD_SET'):
            # It's an iterable
            n_args = instruction.argval
            arg = deque()
            for i in range(n_args):
                arg.appendleft(args.pop())
                instructions.pop(idx - 1)
                idx -= 1
            arg = list(arg)
            if instruction.opname == 'BUILD_TUPLE':
                arg = tuple(arg)
            elif instruction.opname == 'BUILD_SET':
                arg = set(arg)
            args.append(arg)
        elif instruction.opname == "BUILD_MAP":
            # It's a map
            n_keys = instruction.argval
            arg = {}
            for i in range(n_keys):
                v = args.pop()
                k = args.pop()
                instructions.pop(idx - 1)
                idx -= 1
                instructions.pop(idx - 1)
                idx -= 1
                arg[k] = v
            args.append(arg)
        elif instruction.opname == "BUILD_CONST_KEY_MAP":
            # It's a map that had constant keys
            keys = args.pop()
            instructions.pop(idx - 1)
            idx -= 1
            vals = deque()
            for i in range(instruction.argval):
                vals.appendleft(args.pop())
                instructions.pop(idx - 1)
                idx -= 1
            args.append({key: val for key, val in zip(keys, vals)})
        elif instruction.opname == 'LOAD_DEREF' and not _deref_is_callable(
                instruction, closure_vars) and not instructions[idx + 1].opname in ('LOAD_METHOD', 'LOAD_ATTR'):
            # It's a reference to a variable that's not being used to invoke some other function
            args.append(
                closure_vars.nonlocals.get(
                    instruction.argval,
                    closure_vars.globals.get(instruction.argval, closure_vars.builtins.get(instruction.argval, None))))
        elif instruction.opname in ('LOAD_METHOD', 'LOAD_ATTR', 'LOAD_GLOBAL', 'LOAD_DEREF'):
            # We're setting up a function call, which may or may not be invoked
            # Look ahead to combine all of the function pieces together into 1 variable
            name = instructions[idx].argval
            func_pair = _Function(
                closure_vars.nonlocals.get(name, closure_vars.globals.get(name, closure_vars.builtins.get(name, None))),
                name=name)
            if func_pair.func is None:
                # This function can't be found for some reason
                return _parse_lambda_fallback(function)
            while idx + 1 < len(instructions):
                if instructions[idx + 1].opname in ('LOAD_METHOD', 'LOAD_ATTR'):
                    name = instructions[idx + 1].argval
                    func_pair = _Function(getattr(func_pair.func, name), name=func_pair.name + f".{name}")
                    instructions.pop(idx + 1)
                else:
                    break
            args.append(func_pair)
        elif instruction.opname in ('CALL_METHOD', 'CALL_FUNCTION', 'CALL_FUNCTION_KW'):
            kwargs = {}
            kwarg_names = []
            if instruction.opname == 'CALL_FUNCTION_KW':
                # Gather the keywords, which were added with a LOAD_CONST call
                kwarg_names = args.pop()
                instructions.pop(idx - 1)
                idx -= 1
            # Gather the args
            n_args = instruction.argval
            fn_args = deque()
            for i in range(n_args):
                fn_args.appendleft(args.pop())
                instructions.pop(idx - 1)
                idx -= 1
            for name in reversed(kwarg_names):
                kwargs[name] = fn_args.pop()
            # Gather the fn
            func_pair = args.pop()
            instructions.pop(idx - 1)  # Remove the method def from the stack
            idx -= 1
            # Bind the fn
            if not callable(func_pair.func):
                # This shouldn't ever happen, but just in case...
                return _parse_lambda_fallback(function)
            try:
                bound_args = inspect.signature(func_pair.func).bind(*fn_args, **kwargs)
                bound_args.apply_defaults()
            except ValueError:
                # Some functions (C bindings) don't have convenient signature lookup
                bound_args = _PartialBind(tuple(fn_args), kwargs)
            args.append(_BoundFn(func_pair, bound_args))
        elif instruction.opname.startswith('BINARY_') or instruction.opname.startswith(
                'INPLACE_') or instruction.opname == 'COMPARE_OP':
            # Capture actual inline function stuff like: 0.5 + x
            command = strip_prefix(strip_prefix(instruction.opname, 'BINARY_'), 'INPLACE_')
            if instruction.opname == 'COMPARE_OP':
                command = instruction.argval
            if command not in _CommandTable:
                return _parse_lambda_fallback(function)
            right = args.pop()
            instructions.pop(idx - 1)
            idx -= 1
            left = args.pop()
            instructions.pop(idx - 1)
            idx -= 1
            args.append(_Command(left, right, _CommandTable[command]))
        elif instruction.opname == 'POP_JUMP_IF_FALSE':
            # a if a < b else b     |||     <left> if <condition> else <right>
            conditions.append(_Condition(left=None, right=None, condition=args.pop()))
            instructions.pop(idx - 1)
            idx -= 1
        else:
            # TODO - to be fully rigorous we need the rest: https://docs.python.org/3.7/library/dis.html#bytecodes
            # TODO - LIST_APPEND, SET_ADD, MAP_ADD, BUILD_STRING, CALL_FUNCTION_EX, BUILD_TUPLE_UNPACK, etc.
            # Note that this function is only ever used to examine lambda functions, which helps to restrict the set of
            # possible commands
            return _parse_lambda_fallback(function)
        idx += 1
    # Return the bound args
    if conditions or len(args) != 1:
        return _parse_lambda_fallback(function)
    return args[0]


def trace_model(model: Model, model_idx: int, model_fn: Any, optimizer_fn: Any, weights_path: Any) -> Model:
    """A function to add traceability information to an FE-compiled model.

    Args:
        model: The model to be made traceable.
        model_idx: Which of the return values from the `model_fn` is this model (or -1 if only a single return value).
        model_fn: The function used to generate this model.
        optimizer_fn: The thing used to define this model's optimizer.
        weights_path: The path to the weights for this model.

    Returns:
        The `model`, but now with an fe_summary() method.
    """
    prefix = "the" if model_idx == -1 else "the 1st" if model_idx == 0 else "the 2nd" if model_idx == 1 else "the 3rd" \
        if model_idx == 2 else "the {}th".format(model_idx + 1)
    model_fn_summary = strip_prefix(_trace_value(model_fn), "a lambda function passing no arguments to: ")
    optimizer_fn_summary = " with no optimizer" if not optimizer_fn or isinstance(
        optimizer_fn, list) and optimizer_fn[0] is None else " using an optimizer defined by {}".format(
            strip_prefix(_trace_value(optimizer_fn), "a lambda function passing no arguments to: "))
    weights_suffix = "" if not weights_path else " and weights specified by {}".format(_trace_value(weights_path))
    model._fe_traceability_summary = "{} neural network model ('{}') generated from {}{}{}".format(
        prefix, model.model_name, model_fn_summary, optimizer_fn_summary, weights_suffix)

    def fe_summary(self) -> str:
        """Return a summary of how this class was instantiated (for traceability).

        Args:
            self: The bound class instance.

        Returns:
            A summary of the instance.
        """
        return f"This experiment used {self._fe_traceability_summary}"

    # Use MethodType to bind the method to the class instance
    setattr(model, 'fe_summary', types.MethodType(fe_summary, model))
    return model


def traceable(whitelist: Union[str, Tuple[str]] = (), blacklist: Union[str, Tuple[str]] = ()) -> Callable:
    """A decorator to be placed on classes in order to make them traceable and to enable a deep restore.

    Decorated classes will gain the .fe_summary() and .fe_state() methods.

    Args:
        whitelist: Arguments which should be included in a deep restore of the decorated class.
        blacklist: Arguments which should be excluded from a deep restore of the decorated class.

    Returns:
        The decorated class.
    """
    if isinstance(whitelist, str):
        whitelist = (whitelist, )
    if isinstance(blacklist, str):
        blacklist = (blacklist, )
    if whitelist and blacklist:
        raise ValueError("Traceable objects may specify a whitelist or a blacklist, but not both")

    def make_traceable(cls):
        base_init = getattr(cls, '__init__')
        if hasattr(base_init, '__module__') and base_init.__module__ != 'fastestimator.util.traceability_util':
            # We haven't already overridden this class' init method
            def init(self, *args, **kwargs):
                if not hasattr(self, '_fe_state_whitelist'):
                    self._fe_state_whitelist = whitelist
                if not hasattr(self, '_fe_state_blacklist'):
                    self._fe_state_blacklist = blacklist + (
                        '_fe_state_whitelist', '_fe_state_blacklist', '_fe_base_init')
                if not hasattr(self, '_fe_traceability_summary'):
                    bound_args = inspect.signature(base_init).bind(self, *args, **kwargs)
                    bound_args.apply_defaults()
                    bound_args = bound_args.arguments
                    bound_args.pop('self')
                    self._fe_traceability_summary = _trace_value(_BoundFn(self, bound_args))
                base_init(self, *args, **kwargs)

            setattr(cls, '__init__', init)

        def fe_summary(self) -> str:
            """Return a summary of how this class was instantiated (for traceability).

            Args:
                self: The bound class instance.

            Returns:
                A summary of the instance.
            """
            return f"This experiment used {self._fe_traceability_summary}"

        base_func = getattr(cls, 'fe_summary', None)
        if base_func is None:
            setattr(cls, 'fe_summary', fe_summary)

        def fe_state(self) -> Mapping[str, Any]:
            """Return a summary of this class' state variables (for deep restore).

            Args:
                self: The bound class instance.

            Returns:
                The state variables to be considered for deep restore.
            """
            state_dict = {key: value for key, value in self.__dict__.items()}
            if self._fe_state_whitelist:
                state_dict = {key: state_dict[key] for key in whitelist}
            for key in self._fe_state_blacklist:
                state_dict.pop(key, None)
            return state_dict

        base_func = getattr(cls, 'fe_state', None)
        # If the user specified a whitelist or blacklist then use the default impl
        if whitelist or blacklist or base_func is None:
            setattr(cls, 'fe_state', fe_state)

        return cls

    return make_traceable
