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

import platform
import shutil
import sys

import matplotlib
import torch
from pylatex import Command, Document, Figure, Itemize, LongTable, MultiColumn, NoEscape, Package, Section, escape_latex
from pylatex.utils import bold

import fastestimator as fe
from fastestimator.summary.logs.log_plot import plot_logs
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class Traceability(Trace):
    """Automatically generate summary reports of the training.

    Args:
        save_path: Where to save the output files.
    """
    def __init__(self, save_path: str):
        super().__init__(mode={"train", "eval"})
        self.doc = Document(geometry_options=['lmargin=2cm', 'rmargin=2cm', 'tmargin=2cm', 'bmargin=2cm'])
        self.doc.packages.append(Package(name='placeins', options=['section']))  # Keep tables in their sections
        self.save_path = save_path
        self.config_tables = []

    def on_begin(self, data: Data) -> None:

        exp_name = self.system.summary.name
        if not exp_name:
            raise RuntimeError("Traceability reports require an experiment name to be provided in estimator.fit()")
        self.config_tables = self.system.summary.system_config

        self.doc.preamble.append(NoEscape(r'\maxdeadcycles=' + str(2 * len(self.config_tables) + 10) + ''))
        self.doc.preamble.append(NoEscape(r'\extrafloats{' + str(len(self.config_tables) + 10) + '}'))

        self.doc.preamble.append(Command('title', exp_name))
        self.doc.preamble.append(Command('author', f"FastEstimator {fe.__version__}"))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

    def on_end(self, data: Data) -> None:
        with self.doc.create(Section("Training Graphs")):
            with self.doc.create(Figure(position='h!')) as plot:
                old_backend = matplotlib.get_backend() or 'Agg'
                matplotlib.use('Agg')
                plot_logs(experiments=[self.system.summary])
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)
                matplotlib.use(old_backend)

        with self.doc.create(Section("Initialization Parameters")):
            for tbl in self.config_tables:
                tbl.render_table(self.doc)

        with self.doc.create(Section("System Config")):
            with self.doc.create(Itemize()) as itemize:
                itemize.add_item(escape_latex(f"FastEstimator {fe.__version__}"))
                itemize.add_item(escape_latex(f"Python {platform.python_version()}"))
                itemize.add_item(escape_latex(f"OS: {sys.platform}"))
                itemize.add_item(f"Number of GPUs: {torch.cuda.device_count()}")
                if fe.fe_deterministic_seed is not None:
                    itemize.add_item(escape_latex(f"Deterministic Seed: {fe.fe_deterministic_seed}"))
            with self.doc.create(LongTable('|lr|', pos=['h!'], booktabs=True)) as tabular:
                tabular.add_row((bold("Module"), bold("Version")))
                tabular.add_hline()
                tabular.end_table_header()
                tabular.add_hline()
                tabular.add_row((MultiColumn(2, align='r', data='Continued on Next Page'), ))
                tabular.add_hline()
                tabular.end_table_footer()
                tabular.end_table_last_footer()
                color = True
                for name, module in sorted(sys.modules.items()):
                    if "." in name:
                        continue  # Skip sub-packages
                    if name.startswith("_"):
                        continue  # Skip private packages
                    if hasattr(module, '__version__'):
                        tabular.add_row((escape_latex(name), escape_latex(str(module.__version__))),
                                        color='black!5' if color else 'white')
                        color = not color
                    elif hasattr(module, 'VERSION'):
                        tabular.add_row((escape_latex(name), escape_latex(str(module.VERSION))),
                                        color='black!5' if color else 'white')
                        color = not color
        if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
            # No LaTeX Compiler is available
            self.doc.generate_tex(self.save_path)
            # TODO - Gather the images from their tmp dirs so that user can compile later
        else:
            self.doc.generate_pdf(self.save_path, clean_tex=False)
