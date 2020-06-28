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

import locale
import os
import platform
import shutil
import sys
from collections import defaultdict
from typing import DefaultDict, List, Union
from unittest.mock import Base, MagicMock

import dot2tex as d2t
import jsonpickle
import matplotlib
import pydot
import pytorch_model_summary as pms
import tensorflow as tf
import torch
from natsort import humansorted
from pylatex import Command, Document, Figure, Hyperref, Itemize, Label, LongTable, Marker, MultiColumn, NoEscape, \
    Package, Section, Subsection, Subsubsection, Tabular, escape_latex
from pylatex.base_classes import Arguments
from pylatex.utils import bold
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.dataset.dataset import FEDataset
from fastestimator.estimator import Estimator
from fastestimator.network import BaseNetwork
from fastestimator.op.numpyop.meta.one_of import OneOf
from fastestimator.op.numpyop.meta.sometimes import Sometimes
from fastestimator.op.op import Op
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler, get_current_items, get_signature_epochs
from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.trace.trace import Trace, sort_traces
from fastestimator.util.data import Data
from fastestimator.util.latex_util import AdjustBox, Center, HrefFEID, Verbatim
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import FEID, Suppressor, prettify_metric_name


@traceable()
class Traceability(Trace):
    """Automatically generate summary reports of the training.

    Args:
        save_path: Where to save the output files. Note that this will generate a new folder with the given name, into
            which the report and corresponding graphics assets will be written.

    Raises: 
        OSError: If graphviz is not installed.
    """
    def __init__(self, save_path: str):
        # Verify that graphviz is available on this machine
        try:
            pydot.Dot.create(pydot.Dot())
        except OSError:
            raise OSError(
                "Traceability requires that graphviz be installed. See www.graphviz.org/download for more information.")
        # Verify that the system locale is functioning correctly
        try:
            locale.getlocale()
        except ValueError:
            raise OSError("Your system locale is not configured correctly. On mac this can be resolved by adding \
                'export LC_ALL=en_US.UTF-8' and 'export LANG=en_US.UTF-8' to your ~/.bash_profile")
        super().__init__(inputs="*", mode="!infer")  # Claim wildcard inputs to get this trace sorted last
        # Report assets will get saved into a folder for portability
        path = os.path.normpath(save_path)
        root_dir = os.path.dirname(path)
        report = os.path.basename(path) or 'report'
        report = report.split('.')[0]
        self.save_dir = os.path.join(root_dir, report)
        self.figure_dir = os.path.join(self.save_dir, 'figures')
        self.report_name = report
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        # Other member variables
        self.config_tables = []
        self.doc = Document()

    def on_begin(self, data: Data) -> None:
        exp_name = self.system.summary.name
        if not exp_name:
            raise RuntimeError("Traceability reports require an experiment name to be provided in estimator.fit()")
        self.config_tables = self.system.summary.system_config
        models = self.system.network.models
        n_floats = len(self.config_tables) + len(models)

        self.doc = Document(geometry_options=['lmargin=2cm', 'rmargin=2cm', 'tmargin=2cm', 'bmargin=2cm'])
        # Keep tables/figures in their sections
        self.doc.packages.append(Package(name='placeins', options=['section']))
        self.doc.preamble.append(NoEscape(r'\usetikzlibrary{positioning}'))

        # Fix an issue with too many tables for LaTeX to render
        self.doc.preamble.append(NoEscape(r'\maxdeadcycles=' + str(2 * n_floats + 10) + ''))
        self.doc.preamble.append(NoEscape(r'\extrafloats{' + str(n_floats + 10) + '}'))

        # Manipulate booktab tables so that their horizontal lines don't break
        self.doc.preamble.append(NoEscape(r'\aboverulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\belowrulesep=0ex'))
        self.doc.preamble.append(NoEscape(r'\renewcommand{\arraystretch}{1.2}'))

        self.doc.preamble.append(Command('title', exp_name))
        self.doc.preamble.append(Command('author', f"FastEstimator {fe.__version__}"))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

        # TOC
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.doc.append(NoEscape(r'\newpage'))

    def on_end(self, data: Data) -> None:
        self._document_training_graphs()
        self._document_fe_graph()
        self._document_init_params()
        self._document_models()
        self._document_sys_config()

        # Need to move the tikz dependency after the xcolor package
        self.doc.dumps_packages()
        packages = self.doc.packages
        tikz = Package(name='tikz')
        packages.discard(tikz)
        packages.add(tikz)

        if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
            # No LaTeX Compiler is available
            self.doc.generate_tex(os.path.join(self.save_dir, self.report_name))
        else:
            # Force a double-compile since some compilers will struggle with TOC generation
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False, clean=False)
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False)

    def _document_training_graphs(self) -> None:
        """Add training graphs to the traceability document.
        """
        with self.doc.create(Section("Training Graphs")):
            log_path = os.path.join(self.figure_dir, 'logs.png')
            visualize_logs(experiments=[self.system.summary],
                           save_path=log_path,
                           verbose=False,
                           ignore_metrics={'num_device', 'logging_interval'})
            with self.doc.create(Figure(position='h!')) as plot:
                plot.add_image(os.path.relpath(log_path, start=self.save_dir),
                               width=NoEscape(r'1.0\textwidth,height=0.95\textheight,keepaspectratio'))

    def _document_fe_graph(self) -> None:
        """Add FE execution graphs into the traceability document.
        """
        with self.doc.create(Section("FastEstimator Architecture")):
            for mode in self.system.pipeline.data.keys():
                self.doc.append(NoEscape(r'\FloatBarrier'))
                with self.doc.create(Subsection(mode.capitalize())):
                    scheduled_items = self.system.pipeline.get_scheduled_items(
                        mode) + self.system.network.get_scheduled_items(mode) + self.system.traces
                    signature_epochs = get_signature_epochs(scheduled_items,
                                                            total_epochs=self.system.epoch_idx,
                                                            mode=mode)
                    epochs_with_data = self.system.pipeline.get_epochs_with_data(total_epochs=self.system.epoch_idx,
                                                                                 mode=mode)
                    for epoch in signature_epochs:
                        if epoch not in epochs_with_data:
                            continue
                        self.doc.append(NoEscape(r'\FloatBarrier'))
                        with self.doc.create(
                                Subsubsection(f"Epoch {epoch}",
                                              label=Label(Marker(name=f"{mode}{epoch}", prefix="ssubsec")))):
                            diagram = self._draw_diagram(mode, epoch)
                            ltx = d2t.dot2tex(diagram.to_string(), figonly=True)
                            args = Arguments(**{'max width': r'\textwidth, max height=0.9\textheight'})
                            args.escape = False
                            with self.doc.create(Center()):
                                with self.doc.create(AdjustBox(arguments=args)) as box:
                                    box.append(NoEscape(ltx))

    def _document_init_params(self) -> None:
        """Add initialization parameters to the traceability document.
        """
        with self.doc.create(Section("Parameters")):
            model_ids = {
                FEID(id(model))
                for model in self.system.network.models if isinstance(model, (tf.keras.Model, torch.nn.Module))
            }
            datasets = {
                FEID(id(self.system.pipeline.data.get(title, None))):
                (title, self.system.pipeline.data.get(title, None))
                for title in ['train', 'eval', 'test']
            }
            for tbl in self.config_tables:
                name_override = None
                toc_ref = None
                extra_rows = None
                if issubclass(tbl.type, Estimator):
                    toc_ref = "Estimator"
                if issubclass(tbl.type, BaseNetwork):
                    toc_ref = "Network"
                if issubclass(tbl.type, Pipeline):
                    toc_ref = "Pipeline"
                if tbl.fe_id in model_ids:
                    # Link to a later detailed model description
                    name_override = Hyperref(Marker(name=str(tbl.name), prefix="subsec"),
                                             text=NoEscape(r'\textcolor{blue}{') + bold(tbl.name) + NoEscape('}'))
                    toc_ref = tbl.name
                if tbl.fe_id in datasets:
                    title, dataset = datasets[tbl.fe_id]
                    name_override = bold(f'{tbl.name} ({title.capitalize()})')
                    toc_ref = f"{title.capitalize()} Dataset"
                    # Enhance the dataset summary
                    if isinstance(dataset, FEDataset):
                        extra_rows = list(dataset.summary().__getstate__().items())
                        for idx, (key, val) in enumerate(extra_rows):
                            key = f"{prettify_metric_name(key)}:"
                            if isinstance(val, dict) and val:
                                if isinstance(list(val.values())[0], (int, float, str, bool, type(None))):
                                    val = jsonpickle.dumps(val, unpicklable=False)
                                else:
                                    subtable = Tabular('l|l')
                                    for k, v in val.items():
                                        if hasattr(v, '__getstate__'):
                                            v = jsonpickle.dumps(v, unpicklable=False)
                                        subtable.add_row((k, v))
                                    val = subtable
                            extra_rows[idx] = (key, val)
                tbl.render_table(self.doc, name_override=name_override, toc_ref=toc_ref, extra_rows=extra_rows)

    def _document_models(self) -> None:
        """Add model summaries to the traceability document.
        """
        with self.doc.create(Section("Models")):
            for model in humansorted(self.system.network.models, key=lambda m: m.model_name):
                if not isinstance(model, (tf.keras.Model, torch.nn.Module)):
                    continue
                self.doc.append(NoEscape(r'\FloatBarrier'))
                with self.doc.create(Subsection(f"{model.model_name}")):
                    if isinstance(model, tf.keras.Model):
                        # Text Summary
                        summary = []
                        model.summary(line_length=92, print_fn=lambda x: summary.append(x))
                        summary = "\n".join(summary)
                        self.doc.append(Verbatim(summary))
                        with self.doc.create(Center()):
                            self.doc.append(HrefFEID(FEID(id(model)), model.model_name))

                        # Visual Summary
                        # noinspection PyBroadException
                        try:
                            file_path = os.path.join(self.figure_dir, f"FE_Model_{model.model_name}.pdf")
                            dot = tf.keras.utils.model_to_dot(model, show_shapes=True, expand_nested=True)
                            # LaTeX \maxdim is around 575cm (226 inches), so the image must have max dimension less than
                            # 226 inches. However, the 'size' parameter doesn't account for the whole node height, so
                            # set the limit lower (100 inches) to leave some wiggle room.
                            dot.set('size', '100')
                            dot.write(file_path, format='pdf')
                        except Exception:
                            file_path = None
                            print(
                                f"FastEstimator-Warn: Model {model.model_name} could not be visualized by Traceability")
                    elif isinstance(model, torch.nn.Module):
                        if hasattr(model, 'fe_input_spec'):
                            # Text Summary
                            # noinspection PyUnresolvedReferences
                            inputs = model.fe_input_spec.get_dummy_input()
                            self.doc.append(Verbatim(pms.summary(model, inputs)))
                            with self.doc.create(Center()):
                                self.doc.append(HrefFEID(FEID(id(model)), model.model_name))

                            # Visual Summary
                            # Import has to be done while matplotlib is using the Agg backend
                            old_backend = matplotlib.get_backend() or 'Agg'
                            matplotlib.use('Agg')
                            # noinspection PyBroadException
                            try:
                                # Fake the IPython import when user isn't running from Jupyter
                                sys.modules.setdefault('IPython', MagicMock())
                                sys.modules.setdefault('IPython.display', MagicMock())
                                import hiddenlayer as hl
                                with Suppressor():
                                    graph = hl.build_graph(model, inputs)
                                graph = graph.build_dot()
                                graph.attr(rankdir='TB')  # Switch it to Top-to-Bottom instead of Left-to-Right
                                # LaTeX \maxdim is around 575cm (226 inches), so the image must have max dimension less
                                # than 226 inches. However, the 'size' parameter doesn't account for the whole node
                                # height, so set the limit lower (100 inches) to leave some wiggle room.
                                graph.attr(size="100,100")
                                graph.attr(margin='0')
                                file_path = graph.render(filename=f"FE_Model_{model.model_name}",
                                                         directory=self.figure_dir,
                                                         format='pdf',
                                                         cleanup=True)
                            except Exception:
                                file_path = None
                                print("FastEstimator-Warn: Model {} could not be visualized by Traceability".format(
                                    model.model_name))
                            finally:
                                matplotlib.use(old_backend)
                        else:
                            self.doc.append("This model was not used by the Network during training.")
                    if file_path:
                        with self.doc.create(Figure(position='ht!')) as fig:
                            fig.append(Label(Marker(name=str(FEID(id(model))), prefix="model")))
                            fig.add_image(os.path.relpath(file_path, start=self.save_dir),
                                          width=NoEscape(r'1.0\textwidth,height=0.95\textheight,keepaspectratio'))
                            fig.add_caption(NoEscape(HrefFEID(FEID(id(model)), model.model_name).dumps()))

    def _document_sys_config(self) -> None:
        """Add a system config summary to the traceability document.
        """
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
                for name, module in humansorted(sys.modules.items(), key=lambda x: x[0]):
                    if "." in name:
                        continue  # Skip sub-packages
                    if name.startswith("_"):
                        continue  # Skip private packages
                    if isinstance(module, Base):
                        continue  # Skip fake packages we mocked
                    if hasattr(module, '__version__'):
                        tabular.add_row((escape_latex(name), escape_latex(str(module.__version__))),
                                        color='black!5' if color else 'white')
                        color = not color
                    elif hasattr(module, 'VERSION'):
                        tabular.add_row((escape_latex(name), escape_latex(str(module.VERSION))),
                                        color='black!5' if color else 'white')
                        color = not color

    def _draw_diagram(self, mode: str, epoch: int) -> pydot.Dot:
        """Draw a summary diagram of the FastEstimator Ops / Traces.

        Args:
            mode: The execution mode to summarize ('train', 'eval', 'test', or 'infer').
            epoch: The epoch to summarize.

        Returns:
            A pydot digraph representing the execution flow.
        """
        ds = self.system.pipeline.data[mode]
        if isinstance(ds, Scheduler):
            ds = ds.get_current_value(epoch)
        pipe_ops = get_current_items(self.system.pipeline.ops, run_modes=mode, epoch=epoch) if isinstance(
            ds, Dataset) else []
        net_ops = get_current_items(self.system.network.ops, run_modes=mode, epoch=epoch)
        traces = sort_traces(get_current_items(self.system.traces, run_modes=mode, epoch=epoch))
        diagram = pydot.Dot()
        diagram.set('rankdir', 'TB')
        diagram.set('dpi', 300)
        diagram.set_node_defaults(shape='record')
        diagram.add_node(
            pydot.Node(str(id(ds)),
                       label=f'{ds.__class__.__name__} ({FEID(id(ds))})',
                       texlbl=HrefFEID(FEID(id(ds)), name=ds.__class__.__name__).dumps()))
        label_last_seen = defaultdict(lambda: str(id(ds)))  # Where was this key last generated

        self._draw_subgraph(diagram, label_last_seen, 'Pipeline', pipe_ops)
        self._draw_subgraph(diagram, label_last_seen, 'Network', net_ops)
        self._draw_subgraph(diagram, label_last_seen, 'Traces', traces)
        return diagram

    @staticmethod
    def _draw_subgraph(diagram: pydot.Dot,
                       label_last_seen: DefaultDict[str, str],
                       subgraph_name: str,
                       subgraph_ops: List[Union[Op, Trace]]) -> None:
        """Draw a subgraph of ops into an existing `diagram`.

        Args:
            diagram: The diagram to be appended to.
            label_last_seen: A mapping of {data_dict_key: node_id} indicating the last node which generated the key.
            subgraph_name: The name to be associated with this subgraph.
            subgraph_ops: The ops to be wrapped in this subgraph.
        """
        subgraph = pydot.Cluster(style='dashed', graph_name=subgraph_name)
        subgraph.set('label', subgraph_name)
        subgraph.set('labeljust', 'l')
        for idx, op in enumerate(subgraph_ops):
            node_id = str(id(op))
            Traceability._add_node(subgraph, op, node_id)
            edge_srcs = defaultdict(lambda: [])
            for inp in op.inputs:
                if inp == '*':
                    continue
                edge_srcs[label_last_seen[inp]].append(inp)
            for src, labels in edge_srcs.items():
                diagram.add_edge(pydot.Edge(src=src, dst=node_id, label=f" {', '.join(labels)} "))
            for out in op.outputs:
                label_last_seen[out] = node_id
            if isinstance(op, Trace) and idx > 0:
                # Invisibly connect traces in order so that they aren't all just squashed horizontally into the image
                diagram.add_edge(pydot.Edge(src=str(id(subgraph_ops[idx - 1])), dst=node_id, style='invis'))
        diagram.add_subgraph(subgraph)

    @staticmethod
    def _add_node(diagram: Union[pydot.Dot, pydot.Cluster], op: Union[Op, Trace], node_id: str) -> None:
        """Draw a node onto a diagram based on a given op.

        Args:
            diagram: The diagram to be appended to.
            op: The op (or trace) to be visualized.
            node_id: The id to use as the node label.
        """
        if isinstance(op, Sometimes) and op.numpy_op:
            wrapper = pydot.Cluster(style='loosely dotted', graph_name=str(id(op)))
            wrapper.set('label', f'Sometimes ({op.prob}):')
            wrapper.set('labeljust', 'r')
            Traceability._add_node(wrapper, op.numpy_op, node_id)
            diagram.add_subgraph(wrapper)
        elif isinstance(op, OneOf) and op.numpy_ops:
            wrapper = pydot.Cluster(style='loosely dotted', graph_name=str(id(op)))
            wrapper.set('label', 'One Of:')
            wrapper.set('labeljust', 'r')
            Traceability._add_node(wrapper, op.numpy_ops[0], node_id)
            for sub_op in op.numpy_ops[1:]:
                Traceability._add_node(wrapper, sub_op, str(id(sub_op)))
            diagram.add_subgraph(wrapper)
        else:
            if isinstance(op, ModelOp):
                label = f"{op.__class__.__name__} ({FEID(id(op))}): {op.model.model_name}"
                model_ref = Hyperref(Marker(name=str(op.model.model_name), prefix='subsec'),
                                     text=NoEscape(r'\textcolor{blue}{') + bold(op.model.model_name) +
                                     NoEscape('}')).dumps()
                texlbl = f"{HrefFEID(FEID(id(op)), name=op.__class__.__name__).dumps()}: {model_ref}"
            else:
                label = f"{op.__class__.__name__} ({FEID(id(op))})"
                texlbl = HrefFEID(FEID(id(op)), name=op.__class__.__name__).dumps()
            diagram.add_node(pydot.Node(node_id, label=label, texlbl=texlbl))
