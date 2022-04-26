# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(__name__,
                                            submod_attrs={
                                                'cartesian': ['plot_cartesian', 'visualize_cartesian'],
                                                'heatmap': ['plot_heatmap', 'visualize_heatmap'],
                                                'parallel_coordinate_plot': ['plot_parallel_coordinates',
                                                                             'visualize_parallel_coordinates'],
                                                'visualize': ['visualize_search']
                                            })

if TYPE_CHECKING:
    from fastestimator.search.visualize.cartesian import plot_cartesian, visualize_cartesian
    from fastestimator.search.visualize.heatmap import plot_heatmap, visualize_heatmap
    from fastestimator.search.visualize.parallel_coordinate_plot import plot_parallel_coordinates, \
        visualize_parallel_coordinates
    from fastestimator.search.visualize.visualize import visualize_search
