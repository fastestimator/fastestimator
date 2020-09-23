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
import contextlib
import locale
import os
import platform
import re
import shutil
import sys
import types
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple, Union
from unittest.mock import Base, MagicMock

import dot2tex as d2t
import jsonpickle
import matplotlib
import numpy as np
import pydot
import pytorch_model_summary as pms
import tensorflow as tf
import torch
from natsort import humansorted
from pylatex import Command, Document, Figure, Hyperref, Itemize, Label, LongTable, Marker, MultiColumn, NoEscape, \
    Package, Section, Subsection, Subsubsection, Tabularx, escape_latex
from pylatex.base_classes import Arguments
from pylatex.utils import bold
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.dataset.dataset import FEDataset
from fastestimator.network import BaseNetwork
from fastestimator.op.numpyop.meta.fuse import Fuse
from fastestimator.op.numpyop.meta.one_of import OneOf
from fastestimator.op.numpyop.meta.repeat import Repeat
from fastestimator.op.numpyop.meta.sometimes import Sometimes
from fastestimator.op.op import Op
from fastestimator.op.tensorop.meta.fuse import Fuse as FuseT
from fastestimator.op.tensorop.meta.one_of import OneOf as OneOfT
from fastestimator.op.tensorop.meta.repeat import Repeat as RepeatT
from fastestimator.op.tensorop.meta.sometimes import Sometimes as SometimesT
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler, get_current_items, get_signature_epochs
from fastestimator.summary.logs.log_plot import visualize_logs
from fastestimator.trace.trace import Trace, sort_traces
from fastestimator.util.data import Data
from fastestimator.util.latex_util import AdjustBox, Center, ContainerList, HrefFEID, Verbatim
from fastestimator.util.traceability_util import FeSummaryTable, traceable
from fastestimator.util.util import FEID, LogSplicer, Suppressor, prettify_metric_name, to_list


@traceable()
class Traceability(Trace):
    """Automatically generate summary reports of the training.

    Args:
        save_path: Where to save the output files. Note that this will generate a new folder with the given name, into
            which the report and corresponding graphics assets will be written.
        extra_objects: Any extra objects which are not part of the Estimator, but which you want to capture in the
            summary report. One example could be an extra pipeline which performs pre-processing.

    Raises:
        OSError: If graphviz is not installed.
    """
    def __init__(self, save_path: str, extra_objects: Any = None):
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
        path = os.path.abspath(path)
        root_dir = os.path.dirname(path)
        report = os.path.basename(path) or 'report'
        report = report.split('.')[0]
        self.save_dir = os.path.join(root_dir, report)
        self.resource_dir = os.path.join(self.save_dir, 'resources')
        self.report_name = None  # This will be set later by the experiment name
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.resource_dir, exist_ok=True)
        # Other member variables
        self.config_tables = []
        # Extra objects will automatically get included in the report since this Trace is @traceable, so we don't need
        # to do anything with them. Referencing here to stop IDEs from flagging the argument as unused and removing it.
        to_list(extra_objects)
        self.doc = Document()
        self.log_splicer = None

    def on_begin(self, data: Data) -> None:
        exp_name = self.system.summary.name
        if not exp_name:
            raise RuntimeError("Traceability reports require an experiment name to be provided in estimator.fit()")
        # Convert the experiment name to a report name (useful for saving multiple experiments into same directory)
        report_name = "".join('_' if c == ' ' else c for c in exp_name
                              if c.isalnum() or c in (' ', '_')).rstrip().lower()
        report_name = re.sub('_{2,}', '_', report_name)
        self.report_name = report_name or 'report'
        # Send experiment logs into a file
        log_path = os.path.join(self.resource_dir, f"{report_name}.txt")
        if self.system.mode != 'test':
            # If not running in test mode, we need to remove any old log file since it would get appended to
            with contextlib.suppress(FileNotFoundError):
                os.remove(log_path)
        self.log_splicer = LogSplicer(log_path)
        self.log_splicer.__enter__()
        # Get the initialization summary information for the experiment
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

        self._write_title()
        self._write_toc()

    def on_end(self, data: Data) -> None:
        self._write_body_content()

        # Need to move the tikz dependency after the xcolor package
        self.doc.dumps_packages()
        packages = self.doc.packages
        tikz = Package(name='tikz')
        packages.discard(tikz)
        packages.add(tikz)

        if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
            # No LaTeX Compiler is available
            self.doc.generate_tex(os.path.join(self.save_dir, self.report_name))
            suffix = '.tex'
        else:
            # Force a double-compile since some compilers will struggle with TOC generation
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False, clean=False)
            self.doc.generate_pdf(os.path.join(self.save_dir, self.report_name), clean_tex=False)
            suffix = '.pdf'
        print("FastEstimator-Traceability: Report written to {}{}".format(os.path.join(self.save_dir, self.report_name),
                                                                          suffix))
        self.log_splicer.__exit__()

    def _write_title(self) -> None:
        """Write the title content of the file. Override if you want to build on top of base traceability report.
        """
        self.doc.preamble.append(Command('title', self.system.summary.name))
        self.doc.preamble.append(Command('author', f"FastEstimator {fe.__version__}"))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

    def _write_toc(self) -> None:
        """Write the table of contents. Override if you want to build on top of base traceability report.
        """
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.doc.append(NoEscape(r'\newpage'))

    def _write_body_content(self) -> None:
        """Write the main content of the file. Override if you want to build on top of base traceability report.
        """
        self._document_training_graphs()
        self.doc.append(NoEscape(r'\newpage'))
        self._document_fe_graph()
        self.doc.append(NoEscape(r'\newpage'))
        self._document_init_params()
        self._document_models()
        self._document_sys_config()
        self.doc.append(NoEscape(r'\newpage'))

    def _document_training_graphs(self) -> None:
        """Add training graphs to the traceability document.
        """
        with self.doc.create(Section("Training Graphs")):
            log_path = os.path.join(self.resource_dir, f'{self.report_name}_logs.png')
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
                scheduled_items = self.system.pipeline.get_scheduled_items(
                    mode) + self.system.network.get_scheduled_items(mode) + self.system.traces
                signature_epochs = get_signature_epochs(scheduled_items, total_epochs=self.system.epoch_idx, mode=mode)
                epochs_with_data = self.system.pipeline.get_epochs_with_data(total_epochs=self.system.epoch_idx,
                                                                             mode=mode)
                if set(signature_epochs) & epochs_with_data:
                    self.doc.append(NoEscape(r'\FloatBarrier'))
                    with self.doc.create(Subsection(mode.capitalize())):
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
        from fastestimator.estimator import Estimator  # Avoid circular import
        with self.doc.create(Section("Parameters")):
            model_ids = {
                FEID(id(model))
                for model in self.system.network.models if isinstance(model, (tf.keras.Model, torch.nn.Module))
            }
            # Locate the datasets in order to provide extra details about them later in the summary
            datasets = {}
            for mode in ['train', 'eval', 'test']:
                objs = to_list(self.system.pipeline.data.get(mode, None))
                idx = 0
                while idx < len(objs):
                    obj = objs[idx]
                    if obj:
                        feid = FEID(id(obj))
                        if feid not in datasets:
                            datasets[feid] = ({mode}, obj)
                        else:
                            datasets[feid][0].add(mode)
                    if isinstance(obj, Scheduler):
                        objs.extend(obj.get_all_values())
                    idx += 1
            # Parse the config tables
            start = 0
            start = self._loop_tables(start,
                                      classes=(Estimator, BaseNetwork, Pipeline),
                                      name="Base Classes",
                                      model_ids=model_ids,
                                      datasets=datasets)
            start = self._loop_tables(start,
                                      classes=Scheduler,
                                      name="Schedulers",
                                      model_ids=model_ids,
                                      datasets=datasets)
            start = self._loop_tables(start, classes=Trace, name="Traces", model_ids=model_ids, datasets=datasets)
            start = self._loop_tables(start, classes=Op, name="Ops", model_ids=model_ids, datasets=datasets)
            start = self._loop_tables(start,
                                      classes=(Dataset, tf.data.Dataset),
                                      name="Datasets",
                                      model_ids=model_ids,
                                      datasets=datasets)
            start = self._loop_tables(start,
                                      classes=(tf.keras.Model, torch.nn.Module),
                                      name="Models",
                                      model_ids=model_ids,
                                      datasets=datasets)
            start = self._loop_tables(start,
                                      classes=types.FunctionType,
                                      name="Functions",
                                      model_ids=model_ids,
                                      datasets=datasets)
            start = self._loop_tables(start,
                                      classes=(np.ndarray, tf.Tensor, tf.Variable, torch.Tensor),
                                      name="Tensors",
                                      model_ids=model_ids,
                                      datasets=datasets)
            self._loop_tables(start, classes=Any, name="Miscellaneous", model_ids=model_ids, datasets=datasets)

    def _loop_tables(self,
                     start: int,
                     classes: Union[type, Tuple[type, ...]],
                     name: str,
                     model_ids: Set[FEID],
                     datasets: Dict[FEID, Tuple[Set[str], Any]]) -> int:
        """Iterate through tables grouping them into subsections.

        Args:
            start: What index to start searching from.
            classes: What classes are acceptable for this subsection.
            name: What to call this subsection.
            model_ids: The ids of any known models.
            datasets: A mapping like {ID: ({modes}, dataset)}. Useful for augmenting the displayed information.

        Returns:
            The new start index after traversing as many spaces as possible along the list of tables.
        """
        stop = start
        while stop < len(self.config_tables):
            if classes == Any or issubclass(self.config_tables[stop].type, classes):
                stop += 1
            else:
                break
        if stop > start:
            self.doc.append(NoEscape(r'\FloatBarrier'))
            with self.doc.create(Subsection(name)):
                self._write_tables(self.config_tables[start:stop], model_ids, datasets)
        return stop

    def _write_tables(self,
                      tables: List[FeSummaryTable],
                      model_ids: Set[FEID],
                      datasets: Dict[FEID, Tuple[Set[str], Any]]) -> None:
        """Insert a LaTeX representation of a list of tables into the current doc.

        Args:
            tables: The tables to write into the doc.
            model_ids: The ids of any known models.
            datasets: A mapping like {ID: ({modes}, dataset)}. Useful for augmenting the displayed information.
        """
        for tbl in tables:
            name_override = None
            toc_ref = None
            extra_rows = None
            if tbl.fe_id in model_ids:
                # Link to a later detailed model description
                name_override = Hyperref(Marker(name=str(tbl.name), prefix="subsec"),
                                         text=NoEscape(r'\textcolor{blue}{') + bold(tbl.name) + NoEscape('}'))
            if tbl.fe_id in datasets:
                modes, dataset = datasets[tbl.fe_id]
                title = ", ".join([s.capitalize() for s in modes])
                name_override = bold(f'{tbl.name} ({title})')
                # Enhance the dataset summary
                if isinstance(dataset, FEDataset):
                    extra_rows = list(dataset.summary().__getstate__().items())
                    for idx, (key, val) in enumerate(extra_rows):
                        key = f"{prettify_metric_name(key)}:"
                        if isinstance(val, dict) and val:
                            if isinstance(list(val.values())[0], (int, float, str, bool, type(None))):
                                val = jsonpickle.dumps(val, unpicklable=False)
                            else:
                                subtable = Tabularx('l|X', width_argument=NoEscape(r'\linewidth'))
                                for k, v in val.items():
                                    if hasattr(v, '__getstate__'):
                                        v = jsonpickle.dumps(v, unpicklable=False)
                                    subtable.add_row((k, v))
                                # To nest TabularX, have to wrap it in brackets
                                subtable = ContainerList(data=[NoEscape("{"), subtable, NoEscape("}")])
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
                            file_path = os.path.join(self.resource_dir,
                                                     "{}_{}.pdf".format(self.report_name, model.model_name))
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
                            self.doc.append(
                                Verbatim(
                                    pms.summary(model.module if self.system.num_devices > 1 else model,
                                                inputs,
                                                print_summary=False)))
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
                                    graph = hl.build_graph(model.module if self.system.num_devices > 1 else model,
                                                           inputs)
                                graph = graph.build_dot()
                                graph.attr(rankdir='TB')  # Switch it to Top-to-Bottom instead of Left-to-Right
                                # LaTeX \maxdim is around 575cm (226 inches), so the image must have max dimension less
                                # than 226 inches. However, the 'size' parameter doesn't account for the whole node
                                # height, so set the limit lower (100 inches) to leave some wiggle room.
                                graph.attr(size="100,100")
                                graph.attr(margin='0')
                                file_path = graph.render(filename="{}_{}".format(self.report_name, model.model_name),
                                                         directory=self.resource_dir,
                                                         format='pdf',
                                                         cleanup=True)
                            except Exception:
                                file_path = None
                                print("FastEstimator-Warn: Model {} could not be visualized by Traceability".format(
                                    model.model_name))
                            finally:
                                matplotlib.use(old_backend)
                        else:
                            file_path = None
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
        diagram = pydot.Dot(compound='true')  # Compound lets you draw edges which terminate at sub-graphs
        diagram.set('rankdir', 'TB')
        diagram.set('dpi', 300)
        diagram.set_node_defaults(shape='box')

        # Make the dataset the first of the pipeline ops
        pipe_ops.insert(0, ds)
        label_last_seen = defaultdict(lambda: str(id(ds)))  # Where was this key last generated

        batch_size = ""
        if isinstance(ds, Dataset) and not isinstance(ds, BatchDataset):
            batch_size = self.system.pipeline.batch_size
            if isinstance(batch_size, Scheduler):
                batch_size = batch_size.get_current_value(epoch)
            if batch_size is not None:
                batch_size = f" (Batch Size: {batch_size})"
        self._draw_subgraph(diagram, diagram, label_last_seen, f'Pipeline{batch_size}', pipe_ops)
        self._draw_subgraph(diagram, diagram, label_last_seen, 'Network', net_ops)
        self._draw_subgraph(diagram, diagram, label_last_seen, 'Traces', traces)
        return diagram

    @staticmethod
    def _draw_subgraph(progenitor: pydot.Dot,
                       diagram: Union[pydot.Dot, pydot.Cluster],
                       label_last_seen: DefaultDict[str, str],
                       subgraph_name: str,
                       subgraph_ops: List[Union[Op, Trace, Any]]) -> None:
        """Draw a subgraph of ops into an existing `diagram`.

        Args:
            progenitor: The very top level diagram onto which Edges should be written.
            diagram: The diagram into which to add new Nodes.
            label_last_seen: A mapping of {data_dict_key: node_id} indicating the last node which generated the key.
            subgraph_name: The name to be associated with this subgraph.
            subgraph_ops: The ops to be wrapped in this subgraph.
        """
        subgraph = pydot.Cluster(style='dashed', graph_name=subgraph_name, color='black')
        subgraph.set('label', subgraph_name)
        subgraph.set('labeljust', 'l')
        for idx, op in enumerate(subgraph_ops):
            node_id = str(id(op))
            Traceability._add_node(progenitor, subgraph, op, label_last_seen)
            if isinstance(op, Trace) and idx > 0:
                # Invisibly connect traces in order so that they aren't all just squashed horizontally into the image
                progenitor.add_edge(pydot.Edge(src=str(id(subgraph_ops[idx - 1])), dst=node_id, style='invis'))
        diagram.add_subgraph(subgraph)

    @staticmethod
    def _add_node(progenitor: pydot.Dot,
                  diagram: Union[pydot.Dot, pydot.Cluster],
                  op: Union[Op, Trace, Any],
                  label_last_seen: DefaultDict[str, str],
                  edges: bool = True) -> None:
        """Draw a node onto a diagram based on a given op.

        Args:
            progenitor: The very top level diagram onto which Edges should be written.
            diagram: The diagram to be appended to.
            op: The op (or trace) to be visualized.
            label_last_seen: A mapping of {data_dict_key: node_id} indicating the last node which generated the key.
            edges: Whether to write Edges to/from this Node.
        """
        node_id = str(id(op))
        if isinstance(op, (Sometimes, SometimesT)) and op.op:
            wrapper = pydot.Cluster(style='dotted', color='red', graph_name=str(id(op)))
            wrapper.set('label', f'Sometimes ({op.prob}):')
            wrapper.set('labeljust', 'l')
            edge_srcs = defaultdict(lambda: [])
            if op.extra_inputs:
                for inp in op.extra_inputs:
                    if inp == '*':
                        continue
                    edge_srcs[label_last_seen[inp]].append(inp)
            Traceability._add_node(progenitor, wrapper, op.op, label_last_seen)
            diagram.add_subgraph(wrapper)
            dst_id = Traceability._get_all_nodes(wrapper)[0].get_name()
            for src, labels in edge_srcs.items():
                progenitor.add_edge(
                    pydot.Edge(src=src, dst=dst_id, lhead=wrapper.get_name(), label=f" {', '.join(labels)} "))
        elif isinstance(op, (OneOf, OneOfT)) and op.ops:
            wrapper = pydot.Cluster(style='dotted', color='darkorchid4', graph_name=str(id(op)))
            wrapper.set('label', 'One Of:')
            wrapper.set('labeljust', 'l')
            Traceability._add_node(progenitor, wrapper, op.ops[0], label_last_seen, edges=True)
            for sub_op in op.ops[1:]:
                Traceability._add_node(progenitor, wrapper, sub_op, label_last_seen, edges=False)
            diagram.add_subgraph(wrapper)
        elif isinstance(op, (Fuse, FuseT)) and op.ops:
            Traceability._draw_subgraph(progenitor, diagram, label_last_seen, f'Fuse:', op.ops)
        elif isinstance(op, (Repeat, RepeatT)) and op.op:
            wrapper = pydot.Cluster(style='dotted', color='darkgreen', graph_name=str(id(op)))
            wrapper.set('label', f'Repeat:')
            wrapper.set('labeljust', 'l')
            wrapper.add_node(
                pydot.Node(node_id,
                           label=f'{op.repeat if isinstance(op.repeat, int) else "?"}',
                           shape='doublecircle',
                           width=0.1))
            # dot2tex doesn't seem to handle edge color conversion correctly, so have to set hex color
            progenitor.add_edge(pydot.Edge(src=node_id + ":ne", dst=node_id + ":w", color='#006300'))
            Traceability._add_node(progenitor, wrapper, op.op, label_last_seen)
            # Add repeat edges
            edge_srcs = defaultdict(lambda: [])
            for out in op.outputs:
                if out in op.inputs and out not in op.repeat_inputs:
                    edge_srcs[label_last_seen[out]].append(out)
            for inp in op.repeat_inputs:
                edge_srcs[label_last_seen[inp]].append(inp)
            for src, labels in edge_srcs.items():
                progenitor.add_edge(pydot.Edge(src=src, dst=node_id, constraint=False, label=f" {', '.join(labels)} "))
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
            if isinstance(op, (Op, Trace)) and edges:
                # Need the instance check since subgraph_ops might contain a tf dataset or torch dataloader
                Traceability._add_edge(progenitor, op, label_last_seen)

    @staticmethod
    def _add_edge(progenitor: pydot.Dot, op: Union[Trace, Op], label_last_seen: Dict[str, str]):
        """Draw edges into a given Node.

        Args:
            progenitor: The very top level diagram onto which Edges should be written.
            op: The op (or trace) to be visualized.
            label_last_seen: A mapping of {data_dict_key: node_id} indicating the last node which generated the key.
        """
        node_id = str(id(op))
        edge_srcs = defaultdict(lambda: [])
        for inp in op.inputs:
            if inp == '*':
                continue
            edge_srcs[label_last_seen[inp]].append(inp)
        for src, labels in edge_srcs.items():
            progenitor.add_edge(pydot.Edge(src=src, dst=node_id, label=f" {', '.join(labels)} "))
        for out in op.outputs:
            label_last_seen[out] = node_id

    @staticmethod
    def _get_all_nodes(diagram: Union[pydot.Dot, pydot.Cluster]) -> List[pydot.Node]:
        """Recursively search through a `diagram` looking for Nodes.

        Args:
            diagram: The diagram to be inspected.

        Returns:
            All of the Nodes available within this diagram and its child diagrams.
        """
        nodes = diagram.get_nodes()
        for subgraph in diagram.get_subgraphs():
            nodes.extend(Traceability._get_all_nodes(subgraph))
        return nodes
