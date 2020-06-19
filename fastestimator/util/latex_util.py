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
from typing import Union

from pylatex import NoEscape, Package, escape_latex
from pylatex.base_classes import Container, Environment, Options
from pylatex.lists import Enumerate
from pylatex.utils import bold

from fastestimator.util.util import FEID


class ContainerList(Container):
    """A class to expedite combining pieces of latex together.
    """
    def dumps(self) -> str:
        """Get a string representation of this container.

        Returns:
            A string representation of itself.
        """
        return self.dumps_content()


class PyContainer(ContainerList):
    """A class to convert python containers to a LaTeX representation.

    Args:
        data: The python object to be converted to LaTeX.
    """
    def __init__(self, data: Union[list, tuple, set, dict]):
        self.packages.add(Package('enumitem', options='inline'))
        assert isinstance(data, (list, tuple, set, dict)), f"Unacceptable data type for PyContainer: {type(data)}"
        open_char = '[' if isinstance(data, list) else '(' if isinstance(data, tuple) else r'\{'
        close_char = ']' if isinstance(data, list) else ')' if isinstance(data, tuple) else r'\}'
        ltx = Enumerate(options=Options(NoEscape('label={}'), NoEscape('itemjoin={,}')))
        ltx._star_latex_name = True  # Converts this to an inline list
        self.raw_input = data
        if isinstance(data, dict):
            for key, val in data.items():
                ltx.add_item(ContainerList(data=[key, ": ", val]))
        else:
            for val in data:
                ltx.add_item(val)
        super().__init__(data=[NoEscape(open_char), ltx, NoEscape(close_char)])


class Verbatim(Environment):
    """A class to put a string inside the latex verbatim environment.

    Args:
        data: The string to be wrapped.
    """
    def __init__(self, data: str):
        super().__init__(options=None, arguments=None, start_arguments=None, data=NoEscape(data))
        self.content_separator = '\n'


class Center(Environment):
    """A class to center content in a page.
    """
    pass


class AdjustBox(Environment):
    """A class to adjust the size of boxes.
    """
    packages = [Package('adjustbox')]


class HrefFEID(ContainerList):
    """A class to represent a colored and underlined hyperref based on a given fe_id.

    Args:
        fe_id: The id used to link this hyperref.
        name: A string suffix to be printed as part of the link text.
        link_prefix: The prefix for the hyperlink.
        id_in_name: Whether to include the id in front of the name text.
        bold_name: Whether to bold the name.
    """
    def __init__(self,
                 fe_id: FEID,
                 name: str,
                 link_prefix: str = 'tbl',
                 id_in_name: bool = True,
                 bold_name: bool = False):
        self.content_separator = ''
        self.packages.add(Package('hyperref', options='hidelinks'))
        self.packages.add(Package('ulem'))
        self.packages.add(Package('xcolor', options='table'))
        self.fe_id = fe_id
        self.name = name
        data = [
            NoEscape(r'\hyperref['), escape_latex(f"{link_prefix}:"), fe_id, NoEscape(r']{\textcolor{blue}{\uline{')
        ]
        if id_in_name:
            data.append(fe_id)
            if name:
                data.append(": ")
        if name:
            data.append(bold(escape_latex(name)) if bold_name else escape_latex(name))
        data.append(NoEscape("}}}"))
        super().__init__(data=data)
