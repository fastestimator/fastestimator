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
import functools
import inspect
import re
import sys
import types
from collections import ChainMap, deque, namedtuple
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from pylatex import Document, Label, Marker, MultiColumn, NoEscape, Package, Table, Tabularx, TextColor
from pylatex.base_classes import LatexObject
from pylatex.utils import bold, escape_latex, italic

from fastestimator.backend._to_shape import to_shape
from fastestimator.backend._to_type import to_type
from fastestimator.summary.summary import ValWithError
from fastestimator.util.base_util import strip_prefix, FEID, Flag
from fastestimator.util.latex_util import ContainerList, HrefFEID, PyContainer

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
# If a collection (list, tuple, set, dict) has more than this many entries, its summary will be truncated
_CollectionSizeLimit = 42
# The data types that may be restored by the default __getstate__ method. These are not kept inside the is_restorable()
# method due to a formatting issue, though by externalizing them an end user also has more control if they want to add
# something that was missed.
_RestorableClasses = (int,
                      float,
                      bool,
                      str,
                      type(None),
                      ValWithError,
                      tf.Tensor,
                      tf.Variable,
                      torch.Tensor,
                      np.ndarray,
                      np.number,
                      np.bool_,
                      np.flexible,
                      pd.DataFrame)

Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


class FeInputSpec:
    """A class to keep track of a model's input so that fake inputs can be generated.

    This class is intentionally not @traceable.

    Args:
        model_input: The input to the model.
        model: The model which corresponds to the given `model_input`.
    """
    def __init__(self, model_input: Any, model: Model):
        self.shape = to_shape(model_input)
        self.dtype = to_type(model_input)
        self.device = self._get_device(model_input)
        self.tensor_func = tf.ones if isinstance(model, tf.keras.Model) else torch.ones

    def _get_device(self, data: Any) -> Union[None, str, torch.device]:
        """Get the device on which a tensor or collection of tensors is residing.

        Args:
            data: A tensor or collection of tensors.

        Returns:
            The device on which the tensors are residing
        """
        if tf.is_tensor(data) or isinstance(data, torch.Tensor):
            return data.device
        elif isinstance(data, dict):
            return self._get_device(list(data.values()))
        elif isinstance(data, (list, tuple, set)):
            for val in data:
                device = self._get_device(val)
                if device is not None:
                    return device
        else:
            return None

    def get_dummy_input(self) -> Any:
        """Get fake input for the model.

        Returns:
            Input of the correct shape and dtype for the model.
        """
        return self._from_shape_and_type(self.shape, self.dtype)

    def _from_shape_and_type(self, shape: Any, dtype: Any) -> Any:
        """Constructs tensor(s) with the specified shape and dtype.

        It is assumed that the `shape` and `dtype` arguments have the same container structure. That is to say, if
        `shape` is a list of 5 elements, it is required that `dtype` also be a list of 5 elements.

        Args:
            shape: A shape or (possibly nested) container of shapes.
            dtype: A dtype or (possibly nested) container of dtypes.

        Returns:
            A tensor or collection of tensors corresponding to the shape and dtype arguments.
        """
        if isinstance(dtype, dict):
            return {key: self._from_shape_and_type(value, dtype[key]) for key, value in shape.items()}
        elif isinstance(dtype, list):
            return [self._from_shape_and_type(shape[i], dtype[i]) for i in range(len(shape))]
        elif isinstance(dtype, tuple):
            return tuple([self._from_shape_and_type(shape[i], dtype[i]) for i in range(len(shape))])
        elif isinstance(dtype, set):
            return set([self._from_shape_and_type(s, t) for s, t in zip(shape, dtype)])
        else:
            retval = self.tensor_func(shape, dtype=dtype)
            if isinstance(self.device, torch.device):
                retval = retval.to(self.device)
            return retval


class FeSplitSummary(LatexObject):
    """A class to summarize splits performed on an FE Dataset.

    This class is intentionally not @traceable.
    """
    def __init__(self):
        super().__init__()
        self.data = []

    def add_split(self, parent: Union[FEID, str], fraction: str, seed: Optional[int], stratify: Optional[str]) -> None:
        """Record another split on this dataset.

        Args:
            parent: The id of the parent involved in the split (or 'self' if you are the parent).
            fraction: The string representation of the split fraction that was used.
            seed: The random seed used during the split.
            stratify: The stratify key used during the split.
        """
        self.data.append((parent, fraction, seed, stratify))

    def dumps(self) -> str:
        """Generate a LaTeX formatted representation of this object.

        Returns:
            A LaTeX string representation of this object.
        """
        return " $\\rightarrow$ ".join([
            f"{HrefFEID(parent, name='').dumps() if isinstance(parent, FEID) else parent}({escape_latex(fraction)}" +
            (f", seed={seed}" if seed is not None else "") +
            (f", stratify=`{escape_latex(stratify)}'" if stratify is not None else "") + ")" for parent,
            fraction,
            seed,
            stratify in self.data
        ])


class FeSummaryTable:
    """A class containing summaries of traceability information.

    This class is intentionally not @traceable.

    Args:
        name: The string to be used as the title line in the summary table.
        fe_id: The id of this table, used for cross-referencing from other tables.
        target_type: The type of the object being summarized.
        path: The import path of the object in question. Might be more complicated when methods/functions are involved.
        kwargs: The keyword arguments used to instantiate the object being summarized.
        **fields: Any other information about the summarized object / function.
    """
    name: Union[str, LatexObject]
    fe_id: FEID
    fields: Dict[str, Any]

    def __init__(self,
                 name: str,
                 fe_id: FEID,
                 target_type: Type,
                 path: Union[None, str, LatexObject] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 **fields: Any):
        self.name = name
        self.fe_id = fe_id
        self.type = target_type
        self.path = path
        self.args = fields.pop("args", None)
        self.kwargs = kwargs or {}
        self.fields = fields

    def render_table(self,
                     doc: Document,
                     name_override: Optional[LatexObject] = None,
                     toc_ref: Optional[str] = None,
                     extra_rows: Optional[List[Tuple[str, Any]]] = None) -> None:
        """Write this table into a LaTeX document.

        Args:
            doc: The LaTeX document to be appended to.
            name_override: An optional replacement for this table's name field.
            toc_ref: A reference to be added to the table of contents.
            extra_rows: Any extra rows to be added to the table before the kwargs.
        """
        with doc.create(Table(position='htp!')) as table:
            table.append(NoEscape(r'\refstepcounter{table}'))
            table.append(Label(Marker(name=str(self.fe_id), prefix="tbl")))
            if toc_ref:
                table.append(NoEscape(r'\addcontentsline{toc}{subsection}{' + escape_latex(toc_ref) + '}'))
            with doc.create(Tabularx('|lX|', booktabs=True)) as tabular:
                package = Package('xcolor', options='table')
                if package not in tabular.packages:
                    # Need to invoke a table color before invoking TextColor (bug?)
                    tabular.packages.append(package)
                package = Package('seqsplit')
                if package not in tabular.packages:
                    tabular.packages.append(package)
                tabular.add_row((name_override if name_override else bold(self.name),
                                 MultiColumn(size=1, align='r|', data=TextColor('blue', self.fe_id))))
                tabular.add_hline()
                type_str = f"{self.type}"
                match = re.fullmatch(r'^<.* \'(?P<typ>.*)\'>$', type_str)
                type_str = match.group("typ") if match else type_str
                tabular.add_row(("Type: ", escape_latex(type_str)))
                if self.path:
                    if isinstance(self.path, LatexObject):
                        tabular.add_row(("", self.path))
                    else:
                        tabular.add_row(("", escape_latex(self.path)))
                for k, v in self.fields.items():
                    tabular.add_hline()
                    tabular.add_row((f"{k.capitalize()}: ", v))
                if self.args:
                    tabular.add_hline()
                    tabular.add_row(("Args: ", self.args))
                if extra_rows:
                    for (key, val) in extra_rows:
                        tabular.add_hline()
                        tabular.add_row(key, val)
                if self.kwargs:
                    tabular.add_hline()
                    for idx, (kwarg, val) in enumerate(self.kwargs.items()):
                        tabular.add_row((italic(kwarg), val), color='white' if idx % 2 else 'black!5')


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


def _trace_value(inp: Any, tables: Dict[FEID, FeSummaryTable], ret_ref: Flag, wrap_str: bool = True) -> Any:
    """Convert an input value to a FESummaryTable table representation

    Args:
        inp: The input value to be converted.
        tables: A collection of tables representing objects which are used by the current stack of inputs.
        ret_ref: A flag to indicate that _trace_value is returning a reference (this is used to figure out whether
            functions can be in-lined or deserve their own tables).
        wrap_str: Whether literal string values should be wrapped inside extra quote marks.

    Returns:
        An FESummaryTable representation of the input.
    """
    if isinstance(inp, str):
        inp = f"`{escape_latex(inp)}'" if wrap_str else escape_latex(inp)
        if wrap_str:
            # Prevent extremely long strings from overflowing the table
            return NoEscape(r'\seqsplit{' + inp + '}')
        return inp
    elif isinstance(inp, (int, float, bool, type(None), HrefFEID, FEID, PyContainer, np.number)):
        if isinstance(inp, (int, float, np.number)):
            # Prevent extremely long numbers from overflowing the table
            inp = str(inp)
            if len(inp) < 2:
                # Seqsplit doesn't wrap properly with single character inputs
                return NoEscape(r'\seqsplit{\thinspace ' + inp + '}')
            return NoEscape(r'\seqsplit{' + inp + '}')
        return inp
    elif hasattr(inp, '_fe_traceability_summary'):
        # The first time a traceable object goes through here it won't have it's summary instantiated yet, so it will
        # fall through to the class check at the end to get it's id.
        # noinspection PyProtectedMember,PyUnresolvedReferences
        tables.update(inp._fe_traceability_summary)
        inp_id = FEID(id(inp))
        ret_ref.set_true()
        return HrefFEID(inp_id, tables[inp_id].name)
    elif inspect.ismethod(inp):
        parent = _trace_value(inp.__self__, tables, ret_ref, wrap_str)
        return ContainerList(data=[parent, escape_latex(f".{inp.__name__}")])
    elif inspect.isfunction(inp) or inspect.isclass(inp):
        inp_id = FEID(id(inp))
        if inp_id in tables:
            name = tables[inp_id].name
        else:
            if inspect.isfunction(inp) and inp.__name__ == "<lambda>":
                code = inp.__code__
                var_names = code.co_varnames
                # Attempt to figure out what the lambda function is doing. If it is being used only to invoke some other
                # function (like one might do with LRScheduler), then the parse should work.
                flag = Flag()
                func_description = _parse_lambda(inp, tables, flag) or {}
                func_description['vars'] = _trace_value(var_names, tables, flag, wrap_str=False)
                name = "lambda"
                path = None
                if not flag and func_description.keys() == {'vars', 'function'}:
                    # This is a simple lambda function, so inline it instead of making a new table
                    raw_vars = func_description['vars'].raw_input
                    formatted_vars = []
                    for var in raw_vars:
                        formatted_vars.append(var)
                        formatted_vars.append(', ')
                    if formatted_vars:
                        formatted_vars.pop()  # remove trailing comma
                    return ContainerList(data=[
                        TextColor('cyan', f"{name} "), *formatted_vars, ": ", func_description.get('function', '')
                    ])
            else:
                name = inp.__name__
                path = f"{inp.__module__}.{inp.__qualname__}"
                func_description = {}
            tables[inp_id] = FeSummaryTable(name=name,
                                            fe_id=inp_id,
                                            target_type=type(inp),
                                            path=path,
                                            **func_description)
        ret_ref.set_true()
        return HrefFEID(inp_id, name)
    elif isinstance(inp, _Function):
        inp_id = FEID(id(inp))
        if inp_id not in tables:
            if inspect.ismethod(inp.func):
                path = _trace_value(inp.func, tables, ret_ref, wrap_str)
            elif hasattr(inp.func, '__module__') and hasattr(inp.func, '__qualname__'):
                path = f"{inp.func.__module__}.{inp.func.__qualname__}"
            else:
                path = None
            tables[inp_id] = FeSummaryTable(name=inp.name, fe_id=inp_id, target_type=type(inp.func), path=path)
        ret_ref.set_true()
        return HrefFEID(inp_id, inp.name)
    elif isinstance(inp, _PartialBind):
        return {
            "args": _trace_value(inp.args, tables, ret_ref, wrap_str=True),
            "kwargs": _trace_value(inp.kwargs, tables, ret_ref, wrap_str).raw_input  # unwrap kwargs back into a dict
        }
    elif isinstance(inp, _Command):
        return ContainerList(data=[
            _trace_value(inp.left, tables, ret_ref, wrap_str),
            escape_latex(inp.command),
            _trace_value(inp.right, tables, ret_ref, wrap_str)
        ])
    elif isinstance(inp, _Condition):
        return ContainerList(data=[
            _trace_value(inp.left, tables, ret_ref, wrap_str),
            " if ",
            _trace_value(inp.condition, tables, ret_ref, wrap_str),
            " else ",
            _trace_value(inp.right, tables, ret_ref, wrap_str)
        ])
    elif isinstance(inp, _BoundFn):
        flag = Flag()
        args = _trace_value(inp.args, tables, flag, wrap_str=False)
        kwargs = {}
        if isinstance(inp.args, _PartialBind):
            kwargs = args["kwargs"]
            args = args["args"]
        elif isinstance(args, dict):
            kwargs = args
            args = None
        if not flag and isinstance(inp.func, _Function):
            # The function args are simple, so inline this function in whatever is above it
            if isinstance(args, PyContainer):
                args = args.raw_input
            if isinstance(kwargs, PyContainer):
                kwargs = kwargs.raw_input
            formatted = ["("]
            args = args or ()
            kwargs = kwargs or {}
            for arg in args:
                formatted.append(arg)
                formatted.append(", ")
            for key, value in kwargs.items():
                formatted.append(key)
                formatted.append("=")
                formatted.append(value)
                formatted.append(", ")
            if len(formatted) > 1:
                formatted.pop()  # Remove trailing comma
            formatted.append(")")
            if inspect.ismethod(inp.func.func):
                container_list = _trace_value(inp.func.func, tables, ret_ref, wrap_str)
                container_list.data.extend(formatted)
                return container_list
            return ContainerList(data=[inp.func.name, *formatted])
        else:
            # The function args are complicated, so use the normal approach
            func_href = _trace_value(inp.func, tables, ret_ref, wrap_str)
            inp_id = func_href.fe_id
            inp_table = tables[inp_id]
            inp_table.args = args
            inp_table.kwargs = kwargs
            ret_ref.set_true()
            return func_href
    elif isinstance(inp, inspect.BoundArguments):
        args = inp.arguments
        args.pop('self', None)
        return _trace_value(args, tables, ret_ref, wrap_str=False).raw_input  # unwrap kwargs back into a dict
    elif isinstance(inp, _VarWrap):
        return inp.var
    elif isinstance(inp, (tf.keras.Model, torch.nn.Module)):
        # FE models should never actually get here since they are given summaries by trace_model() during fe.build()
        inp_id = FEID(id(inp))
        if inp_id in tables:
            name = tables[inp_id].name
        else:
            name = inp.model_name if hasattr(inp, 'model_name') else "<Unknown Model Name>"
            tables[inp_id] = FeSummaryTable(name=name, fe_id=inp_id, target_type=type(inp))
        ret_ref.set_true()
        return HrefFEID(inp_id, name)
    elif isinstance(inp, list):
        # For list, tuple, and set, limit tracing to the first limit+1 elements so that constructing objects with huge
        #  collections as input arguments doesn't take unnecessary long (ex. a NumpyDataset built using lists). Use N+1
        #  rather than N so that the report shows an ellipse when it trucates the input.
        return PyContainer(data=[_trace_value(x, tables, ret_ref, wrap_str) for x in inp[:_CollectionSizeLimit + 1]],
                           truncate=_CollectionSizeLimit)
    elif isinstance(inp, tuple):
        return PyContainer(
            data=tuple([_trace_value(x, tables, ret_ref, wrap_str) for x in inp[:_CollectionSizeLimit + 1]]),
            truncate=_CollectionSizeLimit)
    elif isinstance(inp, set):
        return PyContainer(
            data=set([_trace_value(x, tables, ret_ref, wrap_str) for x in islice(inp, _CollectionSizeLimit + 1)]),
            truncate=_CollectionSizeLimit)
    elif isinstance(inp, dict):
        return PyContainer(
            data={
                _trace_value(k, tables, ret_ref, wrap_str=wrap_str): _trace_value(v, tables, ret_ref, wrap_str=True)
                for k,
                v in inp.items()
            },
            truncate=_CollectionSizeLimit)
    elif isinstance(inp, (tf.Tensor, torch.Tensor, np.ndarray, tf.Variable)):
        inp_type = type(inp)
        inp_id = FEID(id(inp))
        if inp_id not in tables:
            if isinstance(inp, (tf.Tensor, torch.Tensor, tf.Variable)):
                if isinstance(inp, torch.Tensor):
                    inp = inp.cpu().detach()
                    inp.numpy()
                # In the elif here we're sure to be tf
                elif inp.dtype != tf.dtypes.variant:
                    inp = inp.numpy()  # The variant dtype can't be cast to numpy()
            rank = inp.ndim
            description = {'shape': inp.shape}
            if rank == 0 or (rank == 1 and inp.shape[0] <= 10):
                description['values'] = str(inp)
            tables[inp_id] = FeSummaryTable(name="tensor", fe_id=inp_id, target_type=inp_type, **description)
        ret_ref.set_true()
        return HrefFEID(inp_id, "tensor")
    # This should be the last elif
    elif hasattr(inp, '__class__'):
        inp_id = FEID(id(inp))
        if inp_id not in tables:
            kwargs = {}
            path = None
            if hasattr(inp, '__dict__') and '_fe_state_whitelist' not in inp.__dict__:
                # Prevent circular recursion
                tables[inp_id] = FeSummaryTable(name=inp.__class__.__name__, target_type=type(inp), fe_id=inp_id)
                # This object isn't @traceable but does have some stored variables that we can summarize.
                kwargs = _trace_value({k: v
                                       for k, v in inp.__dict__.items() if not k.startswith("_")},
                                      tables,
                                      ret_ref,
                                      wrap_str=False).raw_input
                path = "Not @traceable, so summary is approximate"
            tables[inp_id] = FeSummaryTable(name=inp.__class__.__name__,
                                            target_type=type(inp),
                                            path=path,
                                            fe_id=inp_id,
                                            kwargs=kwargs)
        ret_ref.set_true()
        return HrefFEID(inp_id, inp.__class__.__name__)
    else:
        inp_id = FEID(id(inp))
        if inp_id not in tables:
            tables[inp_id] = FeSummaryTable(name="an object", target_type=type(inp), fe_id=inp_id)
        ret_ref.set_true()
        return HrefFEID(inp_id, "an object")


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
            # This lambda function runs to the end of the line, so return even though end_char wasn't found.
            # include_last is ignored in this case since end_char isn't detected.
            return chunk_idx - 1, len(chunks[-1])
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
                    # In a situation with nested lambda functions, find the closest preceding one to own this string
                    owned = True
                    for other in specs:
                        if other.chunk_end < i:
                            continue  # Other cannot contain this chunk
                        if spec.chunk_start < other.chunk_start <= i:
                            owned = False
                            break
                        elif other.chunk_start == spec.chunk_start and other.idx_start > spec.idx_start:
                            owned = False
                            break
                    if owned:
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
            arg = arg.strip()
            if arg:
                results.add(arg)
            arg = ''
        elif left_side:
            arg += char
    return results


def _parse_lambda_fallback(function: types.FunctionType, tables: Dict[FEID, FeSummaryTable],
                           ret_ref: Flag) -> Optional[Dict[str, Any]]:
    """Convert a lambda function into a string representation, disambiguating variables when possible.

    Args:
        function: The lambda function to be inspected.
        tables: A collection of tables representing objects which are used by the current stack of inputs.
        ret_ref: A flag to indicate that _trace_value is returning a reference (this is used to figure out whether
            functions can be in-lined or deserve their own tables).


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
        lambda_fns = list(filter(lambda x: x[1] == strings, lambda_fns))
        if len(lambda_fns) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Throw out any functions which don't contain the right references
        names = set(code.co_names)
        lambda_fns = list(filter(lambda x: all(s in x[2] for s in names), lambda_fns))
        if len(lambda_fns) == 1:
            lambda_fn = lambda_fns[0]
    if lambda_fn is None:
        # Throw out any functions which don't contain the right free variables
        names = set(code.co_freevars)
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
    response = {"function": escape_latex(lam)}
    if refs:
        closure_vars = inspect.getclosurevars(function)
        ref_map = {
            ref:
                closure_vars.nonlocals.get(ref,
                                           closure_vars.globals.get(ref, closure_vars.builtins.get(ref, _VarWrap(ref))))
            for ref in refs
        }
        response['kwargs'] = _trace_value(ref_map, tables, ret_ref=ret_ref, wrap_str=False).raw_input
    return response


def _parse_lambda(function: types.FunctionType, tables: Dict[FEID, FeSummaryTable],
                  ret_ref: Flag) -> Optional[Dict[str, Any]]:
    """Convert a lambda function into its argument-based representation.

    The `function` is expected to be a lambda expression, which means that the set of bytecode instructions is limited
    compared to examining any possible function.

    Args:
        function: A lambda function to be inspected.
        tables: A collection of tables representing objects which are used by the current stack of inputs.
        ret_ref: A flag to indicate that _trace_value is returning a reference (this is used to figure out whether
            functions can be in-lined or deserve their own tables).

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
                return _parse_lambda_fallback(function, tables, ret_ref)
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
                return _parse_lambda_fallback(function, tables, ret_ref)
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
                return _parse_lambda_fallback(function, tables, ret_ref)
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
            return _parse_lambda_fallback(function, tables, ret_ref)
        idx += 1
    # Return the bound args
    if conditions or len(args) != 1:
        return _parse_lambda_fallback(function, tables, ret_ref)
    return {"function": _trace_value(args[0], tables, ret_ref=ret_ref, wrap_str=True)}


def fe_summary(self) -> List[FeSummaryTable]:
    """Return a summary of how this class was instantiated (for traceability).

    Args:
        self: The bound class instance.

    Returns:
        A summary of the instance.
    """
    # Delayed imports to avoid circular dependency
    from torch.utils.data import Dataset

    from fastestimator.estimator import Estimator
    from fastestimator.network import TFNetwork, TorchNetwork
    from fastestimator.op.op import Op
    from fastestimator.pipeline import Pipeline
    from fastestimator.schedule.schedule import Scheduler
    from fastestimator.trace.trace import Trace

    # re-number the references for nicer viewing
    ordered_items = sorted(
        self._fe_traceability_summary.items(),
        key=lambda x: 0 if issubclass(x[1].type, Estimator) else 1
        if issubclass(x[1].type, (TFNetwork, TorchNetwork)) else 2 if issubclass(x[1].type, Pipeline) else 3
        if issubclass(x[1].type, Scheduler) else 4 if issubclass(x[1].type, Trace) else 5
        if issubclass(x[1].type, Op) else 6 if issubclass(x[1].type, (Dataset, tf.data.Dataset)) else 7
        if issubclass(x[1].type, (tf.keras.Model, torch.nn.Module)) else 8
        if issubclass(x[1].type, types.FunctionType) else 9
        if issubclass(x[1].type, (np.ndarray, tf.Tensor, tf.Variable, torch.Tensor)) else 10)
    key_mapping = {fe_id: f"@FE{idx}" for idx, (fe_id, val) in enumerate(ordered_items)}
    FEID.set_translation_dict(key_mapping)
    return [item[1] for item in ordered_items]


def __getstate__(self) -> Dict[str, Any]:
    """Return a summary of this class' state variables (for restore wizard).

    Args:
        self: The bound class instance.

    Returns:
        The state variables to be captured during pickling.
    """
    state_dict = self.__dict__.copy()
    # Classes which subclass a @traceable class but are not themselves marked as @traceable will not inherit their
    # parent whitelist and blacklist. If __getstate__ is being invoked then they'll need at least the basics.
    if not hasattr(self, '_fe_state_whitelist'):
        self._fe_state_whitelist = ()
    if not hasattr(self, '_fe_state_blacklist'):
        self._fe_state_blacklist = ('_fe_state_whitelist', '_fe_state_blacklist', '_fe_traceability_summary')
    # Apply the whitelist and blacklist
    if self._fe_state_whitelist:
        state_dict = {key: state_dict[key] for key in self._fe_state_whitelist}
    for key in self._fe_state_blacklist:
        state_dict.pop(key, None)
    # We can't support complex objects / recursion since lambda functions can't be pickled and collections might
    # be appended to after the init call over the course of training, preventing nested complex objects from
    # being perfectly recovered. The memory limit is to avoid saving a copy of the user's entire dataset if it
    # happens to be held in memory.
    for key, value in list(state_dict.items()):
        keep, size = is_restorable(value, memory_limit=0 if key in self._fe_state_whitelist else 1e6)
        if not keep:
            state_dict.pop(key)
    return state_dict


def __setstate__(self, state: Dict[str, Any]) -> None:
    """Apply saved state information to this class (for restore wizard).

    Args:
        self: The bound class instance.
        state: The new state to be imposed.
    """
    for key, replacement_data in state.items():
        if key not in self.__dict__:
            self.key = replacement_data
            continue
        current_data = self.__dict__[key]
        self.__dict__[key] = _setdata(current_data, replacement_data)


def _setdata(current: Any, new: Any) -> Any:
    """Replace a `current` value with a `new` value, returning whatever value should be used.

    This method helps to perform in-place modifications of values whenever possible so that shared variables are not
    broken by the RestoreWizard.

    Args:
        current: The existing object.
        new: A new value to be imposed on the `current` object.

    Returns:
        The object to be stored in place of the `current` one.
    """
    if isinstance(current, list) and isinstance(new, (tuple, list)) and len(current) == len(new):
        for idx, elem in enumerate(current):
            current[idx] = _setdata(elem, new[idx])
        return current
    # Check for named tuple
    if isinstance(current, tuple) and hasattr(current, '_fields') and hasattr(current, '_make') and \
            isinstance(new, (tuple, list)) and len(current) == len(new):
        return current._make([_setdata(cu, cr) for cu, cr in zip(current, new)])
    if isinstance(current, tuple) and isinstance(new, (tuple, list)) and len(current) == len(new):
        return tuple([_setdata(cu, cr) for cu, cr in zip(current, new)])
    if isinstance(current, dict) and isinstance(new, dict) and current.keys() == new.keys():
        # Might want to consider removing the key equality requirement, but unclear what ramifications that would have.
        for key in current.keys():
            current[key] = _setdata(current[key], new[key])
        return current
    if isinstance(current, tf.Variable) and isinstance(new, tf.Variable) and current.shape == new.shape:
        current.assign(new)
        return current
    if isinstance(current, torch.Tensor) and isinstance(new, torch.Tensor) and current.shape == new.shape:
        current.copy_(new)
        return current
    if isinstance(current, np.ndarray) and isinstance(new, np.ndarray) and current.shape == new.shape:
        np.copyto(dst=current, src=new)
        return current
    if isinstance(new, _RestorableClasses):
        return new  # This should go before __dict__ since tensors have __dict__s
    if hasattr(current, '__setstate__'):
        current.__setstate__(new)
        return current
    if hasattr(current, '__dict__'):
        current.__dict__.update(new)
        return current
    return new


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
    tables = {}
    description = {'definition': _trace_value(model_fn, tables, ret_ref=Flag())}
    if model_idx != -1:
        description['index'] = model_idx
    if optimizer_fn or isinstance(optimizer_fn, list) and optimizer_fn[0] is not None:
        description['optimizer'] = _trace_value(
            optimizer_fn[model_idx] if isinstance(optimizer_fn, list) else optimizer_fn, tables, ret_ref=Flag())
    if weights_path:
        description['weights'] = _trace_value(weights_path, tables, ret_ref=Flag())
    fe_id = FEID(id(model))
    tbl = FeSummaryTable(name=model.model_name, fe_id=fe_id, target_type=type(model), **description)
    tables[fe_id] = tbl
    # Have to put this in a ChainMap b/c dict gets put into model._layers automatically somehow
    model._fe_traceability_summary = ChainMap(tables)

    # Use MethodType to bind the method to the class instance
    setattr(model, 'fe_summary', types.MethodType(fe_summary, model))
    return model


def traceable(whitelist: Union[str, Tuple[str, ...]] = (), blacklist: Union[str, Tuple[str, ...]] = ()) -> Callable:
    """A decorator to be placed on classes in order to make them traceable and to enable a deep restore.

    Decorated classes will gain the .fe_summary() and .fe_state() methods.

    Args:
        whitelist: Arguments which should be included in a deep restore of the decorated class.
        blacklist: Arguments which should be excluded from a deep restore of the decorated class.

    Returns:
        The decorated class.
    """
    if isinstance(whitelist, str):
        whitelist = (whitelist,)
    if isinstance(blacklist, str):
        blacklist = (blacklist,)
    if whitelist and blacklist:
        raise ValueError("Traceable objects may specify a whitelist or a blacklist, but not both")

    def make_traceable(cls):
        base_init = getattr(cls, '__init__')
        if hasattr(base_init, '__module__') and base_init.__module__ != 'fastestimator.util.traceability_util':
            # We haven't already overridden this class' init method
            @functools.wraps(base_init)  # to preserve the original class signature
            def init(self, *args, **kwargs):
                if not hasattr(self, '_fe_state_whitelist'):
                    self._fe_state_whitelist = whitelist
                else:
                    self._fe_state_whitelist = tuple(set(self._fe_state_whitelist).union(set(whitelist)))
                if not hasattr(self, '_fe_state_blacklist'):
                    self._fe_state_blacklist = blacklist + (
                        '_fe_state_whitelist', '_fe_state_blacklist', '_fe_traceability_summary')
                else:
                    self._fe_state_blacklist = tuple(set(self._fe_state_blacklist).union(set(blacklist)))
                if not hasattr(self, '_fe_traceability_summary'):
                    bound_args = inspect.signature(base_init).bind(self, *args, **kwargs)
                    bound_args.apply_defaults()
                    tables = {}
                    _trace_value(_BoundFn(self, bound_args), tables, ret_ref=Flag())
                    self._fe_traceability_summary = tables
                base_init(self, *args, **kwargs)

            setattr(cls, '__init__', init)

        base_func = getattr(cls, 'fe_summary', None)
        if base_func is None:
            setattr(cls, 'fe_summary', fe_summary)

        base_func = getattr(cls, '__getstate__', None)
        if base_func is None:
            setattr(cls, '__getstate__', __getstate__)

        base_func = getattr(cls, '__setstate__', None)
        if base_func is None:
            setattr(cls, '__setstate__', __setstate__)

        return cls

    return make_traceable


def is_restorable(data: Any, memory_limit: int = 0) -> Tuple[bool, int]:
    """Determine whether a given object can be restored easily via Pickle.

    Args:
        data: The object in question.
        memory_limit: The maximum memory size (in bytes) to allow for an object (or 0 for no limit).

    Returns:
        (result, memory size) where result is True iff `data` is only comprised of 'simple' objects and does not exceed
        the `memory_limit`. If the result is False, then memory size will be <= the true memory size of the `data`.
    """
    if isinstance(data, _RestorableClasses):
        size = sys.getsizeof(data)
        if isinstance(data, tf.Tensor):
            size = sys.getsizeof(data.numpy())
        elif isinstance(data, torch.Tensor):
            size = data.element_size() * data.nelement()
        return True, size
    elif isinstance(data, dict):
        size = 0
        for key, value in data.items():
            key_stat = is_restorable(key, memory_limit)
            if key_stat[0] is False:
                return False, size
            size += key_stat[1]
            if 0 < memory_limit < size:
                return False, size
            val_stat = is_restorable(value, memory_limit)
            if val_stat[0] is False:
                return False, size
            size += val_stat[1]
            if 0 < memory_limit < size:
                return False, size
        return True, size
    elif isinstance(data, (list, tuple, set)):
        size = 0
        for elem in data:
            elem_stat = is_restorable(elem, memory_limit)
            if elem_stat[0] is False:
                return False, size
            size += elem_stat[1]
            if 0 < memory_limit < size:
                return False, size
        return True, size
    else:
        return False, 0
