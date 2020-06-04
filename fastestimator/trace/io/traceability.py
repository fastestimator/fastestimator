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

from pylatex import Document, NoEscape

from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class Traceability(Trace):
    def on_begin(self, data: Data) -> None:
        # \newcolumntype{R}{>{\raggedleft\arraybackslash}X}
        # %\renewcommand{\tabularxcolumn}[1]{m{#1}}
        config = self.system.summary.system_config
        doc = Document(geometry_options={'lmargin': "1cm", 'rmargin': "1cm"})
        doc.preamble.append(NoEscape(r'\maxdeadcycles=' + str(2 * len(config) + 10) + ''))
        doc.preamble.append(NoEscape(r'\extrafloats{' + str(len(config) + 10) + '}'))
        for tbl in config:
            tbl.render_table(doc)
        doc.generate_pdf('mypdf', clean_tex=False)
        self.system.stop_training = True  # TODO - Remove this
