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
import inspect
import math
import os
from typing import Any, Callable, Dict, Iterable, Optional, Union

import pandas as pd

from fastestimator.dataset.dataset import InMemoryDataset
from fastestimator.util.base_util import to_list
from fastestimator.util.traceability_util import traceable


@traceable()
class CSVDataset(InMemoryDataset):
    """A dataset from a CSV file.

    CSVDataset reads entries from a CSV file, where the first row is the header. The root directory of the csv file
    may be accessed using dataset.parent_path. This may be useful if the csv contains relative path information
    that you want to feed into, say, an ImageReader Op.

    Args:
        file_path: The (absolute) path to the CSV file.
        delimiter: What delimiter is used by the file.
        include_if: An optional filter specifying which rows should be included. This can be a dictionary, for example
            {'mode': 'train', 'type': [0, 1, 2]} in which case only rows which have a 'mode' column value of 'train' AND
            a 'type' column value of either 0, 1, or 2 will be included in this dataset. Alternatively, this can be a
            query string: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#the-query-method, for
            example 'type >= 1'. Finally, it could be a function whose argument(s) correspond to column names and whose
            output is a boolean (include_if=lambda mode: mode in ['eval', 'test']). This last option is very flexible,
            but also slower to execute.
        fill_na: A fill value if data is missing. By default, this will follow pandas convention and use different types
            of NaNs.
        kwargs: Other arguments to be passed through to pandas csv reader function. See the pandas docs for details:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.
    """

    def __init__(self,
                 file_path: str,
                 delimiter: str = ",",
                 include_if: Union[None, Dict[str, Union[Any, Iterable[Any]]], str, Callable[..., bool]] = None,
                 fill_na: Optional[Any] = 'pandas_default',
                 **kwargs) -> None:
        df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        if fill_na is None:
            df = df.fillna(math.nan).replace([math.nan], [None])
        elif fill_na != 'pandas_default':
            df = df.fillna(value=fill_na)
        if include_if is not None:
            if isinstance(include_if, dict):
                for k, v in include_if.items():
                    v = [None] if v is None else to_list(v)
                    df = df[df[k].isin(v)]
            elif isinstance(include_if, str):
                df = df.query(include_if)
            elif hasattr(include_if, "__call__"):
                cols = list(inspect.signature(include_if).parameters.keys())
                for col in cols:
                    if col not in df.columns:
                        raise ValueError(f"The provided filter function requested '{col}' which was not found in the "
                                         f"csv columns: {list(df.columns)}")
                df = df[df[cols].apply(lambda row: include_if(**row), axis=1)]
            else:
                raise ValueError(f"Received an unexpected datatype for include_if: {type(include_if)}")
        self.parent_path = os.path.dirname(file_path)
        super().__init__(df.to_dict(orient='index'))
