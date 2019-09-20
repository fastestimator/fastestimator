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
import matplotlib.pyplot as plt
from fastestimator.estimator.trace import Logger
from fastestimator.visualization.logs import plot_logs


class VisLogger(Logger):
    """A Logger which visualizes to the screen during training
    """
    def __init__(self, vis_steps, **plot_args):
        super().__init__()
        self.vis_steps = vis_steps
        self.plot_args = plot_args
        self.true_persist = False

    def on_begin(self, state):
        self.log_steps = state['log_steps']
        self.true_persist = state['persist_summary']
        self.persist_summary = True
        self._print_message("FastEstimator-Start: step: {}; ".format(state["train_step"]), state)

    def on_batch_end(self, state):
        super().on_batch_end(state)
        if state["mode"] == "train" and state["train_step"] > 0 and state["train_step"] % self.vis_steps == 0:
            fig = plot_logs(self.summary, **self.plot_args)
            plt.draw()
            plt.pause(0.000001)
            plt.close(fig)

    def on_end(self, state):
        self._print_message("FastEstimator-Finish: step: {}; ".format(state["train_step"]), state)
        if self.true_persist:
            state['summary'].merge(self.summary)
        plot_logs(self.summary, **self.plot_args)
        plt.show()
