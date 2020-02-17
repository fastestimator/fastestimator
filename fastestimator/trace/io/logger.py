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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fastestimator.summary import Summary
from fastestimator.summary.logs import plot_logs
from fastestimator.trace import Trace
from fastestimator.util.util import to_list
from fastestimator.xai import fig_to_img, show_image


class Logger(Trace):
    """Logger that prints log. Please don't add this trace into an estimator manually. An estimators will add it
        automatically.
    """
    def __init__(self):
        super().__init__(inputs="*")
        self.log_steps = 0
        self.persist_summary = False
        self.epoch_losses = []
        self.summary = Summary("")

    def on_begin(self, state):
        self.log_steps = state['log_steps']
        self.persist_summary = state['persist_summary']
        self._print_message("FastEstimator-Start: step: {}; ".format(state["train_step"]), state)

    def on_epoch_begin(self, state):
        self.epoch_losses = self.network.epoch_losses

    def on_batch_end(self, state):
        if state["mode"] == "train" and state["train_step"] % self.log_steps == 0:
            self._print_message("FastEstimator-Train: step: {}; ".format(state["train_step"]), state)

    def on_epoch_end(self, state):
        if state["mode"] == "eval":
            self._print_message("FastEstimator-Eval: step: {}; ".format(state["train_step"]), state, True)

    def on_end(self, state):
        self._print_message("FastEstimator-Finish: step: {}; ".format(state["train_step"]), state)
        state['summary'].merge(self.summary)

    def _print_message(self, header, state, log_epoch=False):
        log_message = header
        if log_epoch:
            log_message += "epoch: {}; ".format(state["epoch"])
            if self.persist_summary:
                self.summary.history[state.get("mode", "train")]['epoch'][state["train_step"]] = state["epoch"]
        results = state.maps[0]
        for key, val in results.items():
            if hasattr(val, "numpy"):
                val = val.numpy()
            if self.persist_summary:
                self.summary.history[state.get("mode", "train")][key][state["train_step"]] = val
            if key in self.epoch_losses:
                val = round(val, 7)
            if isinstance(val, np.ndarray):
                log_message += "\n{}:\n{};".format(key, np.array2string(val, separator=','))
            else:
                log_message += "{}: {}; ".format(key, str(val))
        print(log_message)


class VisLogger(Logger):
    """A Logger which visualizes to the screen during training
    """
    def __init__(self, vis_steps=None, show_images=None, **plot_args):
        super().__init__()
        self.vis_steps = vis_steps
        self.plot_args = plot_args
        self.true_persist = False
        self.show_images = to_list(show_images) if show_images else []
        self.images = {}

    def on_begin(self, state):
        self.log_steps = state['log_steps']
        self.true_persist = state['persist_summary']
        self.persist_summary = True
        self._print_message("FastEstimator-Start: step: {}; ".format(state["train_step"]), state)

    def on_batch_end(self, state):
        super().on_batch_end(state)
        change = False
        if self.vis_steps and state[
                "mode"] == "train" and state["train_step"] > 0 and state["train_step"] % self.vis_steps == 0:
            img = self._gen_log_image()
            self.images["logs"] = img
            change = True

        change = self._find_images(state) or change

        if change:
            self._display_images()

    def on_epoch_end(self, state):
        super().on_epoch_end(state)
        change = self._find_images(state)
        if change:
            self._display_images()

    def on_end(self, state):
        self._print_message("FastEstimator-Finish: step: {}; ".format(state["train_step"]), state)
        if self.true_persist:
            state['summary'].merge(self.summary)
        img = self._gen_log_image()
        self.images["logs"] = img
        self._find_images(state)
        self._display_images()
        plt.show()

    def _gen_log_image(self):
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        fig = plot_logs(self.summary, **self.plot_args)
        fig.canvas.draw()
        img = fig_to_img(fig, batch=False)
        matplotlib.use(old_backend)
        return img

    @staticmethod
    def _display_image(img, title):
        # TODO - Figure out how to keep the windows from forcefully overlapping each-other every time they render
        fig = plt.figure(title, clear=True, tight_layout=True)
        ax = fig.add_subplot(111)
        show_image(img, axis=ax)
        fig.canvas.draw()

    def _display_images(self):
        for title, img in self.images.items():
            self._display_image(img=img, title=title)
        plt.pause(1e-6)

    def _find_images(self, state):
        found = False
        for key in self.show_images:
            imgs = state.get(key)
            if imgs is not None:
                for idx, img in enumerate(imgs):
                    img_key = "{}{}".format(key, "_{}".format(idx) if idx > 0 else "")
                    self.images[img_key] = img
                    found = True
        return found
