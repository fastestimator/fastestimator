# FastEstimator

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/icon.png?raw=true" title="we are cool">
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-informational.svg)](LICENSE)
[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=PR-build&job=fastestimator%2Ffastestimator%2Fmaster)](http://jenkins.fastestimator.org:8080/job/fastestimator/job/fastestimator/job/master/)
[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=nightly-build&job=nightly)](http://jenkins.fastestimator.org:8080/job/nightly/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3a46ea86b8f04caab271f2a7bd6f4bd9)](https://www.codacy.com/gh/fastestimator/fastestimator/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fastestimator/fastestimator&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/3a46ea86b8f04caab271f2a7bd6f4bd9)](https://www.codacy.com/gh/fastestimator/fastestimator/dashboard?utm_source=github.com&utm_medium=referral&utm_content=fastestimator/fastestimator&utm_campaign=Badge_Coverage)
[![PyPI version](https://badge.fury.io/py/fastestimator.svg)](https://pypi.org/project/fastestimator/)
[![PyPI stable Download](https://img.shields.io/pypi/dm/fastestimator?label=stable%20downloads&color=16D1B4)](https://pypistats.org/packages/fastestimator)
[![PyPI stable Download](https://img.shields.io/pypi/dm/fastestimator-nightly?label=nightly%20downloads&color=16D1B4)](https://pypistats.org/packages/fastestimator-nightly)

FastEstimator is a high-level deep learning library built on PyTorch. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:

**FastEstimator 2.0** is a major release that streamlines the framework to be **PyTorch-only**, removes TensorFlow dependencies, migrates to [UV](https://docs.astral.sh/uv/) for modern dependency management, and includes updated ops, traces, and tutorials.

For more information, please visit our [website](https://www.fastestimator.org/).

## Support Matrix

| FastEstimator  | Python | PyTorch | CUDA |  Installation Instruction |
| -------------  | ------  | ------- | ---- | ----------- |
| Nightly  | 3.10-3.12  | 2.3.1 | 12.2.2 | master branch |
| 2.0 (latest stable) | 3.10-3.12  | 2.3.1 | 12.2.2 | [r2.0 branch](https://github.com/fastestimator/fastestimator/tree/r2.0) |
| 1.7 | 3.10-3.12  | 2.3.1 | 12.2 | [r1.7 branch](https://github.com/fastestimator/fastestimator/tree/r1.7) |
| 1.6  | 3.8-3.10  | 2.0.1 | 11.8 | [r1.6 branch](https://github.com/fastestimator/fastestimator/tree/r1.6) |
| 1.5  | 3.7-3.9  | 1.10.2 | 11.0 | [r1.5 branch](https://github.com/fastestimator/fastestimator/tree/r1.5) |
| 1.4  | 3.6-3.8  | 1.7.1 | 11.0 | [r1.4 branch](https://github.com/fastestimator/fastestimator/tree/r1.4) |
| 1.3  | 3.6-3.8  | 1.7.1 | 11.0 | [r1.3 branch](https://github.com/fastestimator/fastestimator/tree/r1.3) |
| 1.2  | 3.6-3.8  | 1.7.1 | 11.0 | [r1.2 branch](https://github.com/fastestimator/fastestimator/tree/r1.2) |
| 1.1  | 3.6-3.8  | 1.6.0 | 10.1 | [r1.1 branch](https://github.com/fastestimator/fastestimator/tree/r1.1) |

## Installation

We recommend using [UV](https://docs.astral.sh/uv/) for fast, reliable package management.

### Quick Start with UV

1. Install UV:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install FastEstimator:
    ```bash
    uv pip install fastestimator
    ```

    PyTorch is included automatically:
    - **macOS**: Standard PyTorch with MPS (Apple Silicon GPU) support
    - **Linux/Windows**: CPU-optimized PyTorch build

### For Developers

```bash
git clone https://github.com/fastestimator/fastestimator.git
cd fastestimator
uv sync                        # Install all dependencies
uv sync --extra jupyter        # Include Jupyter notebook support
uv sync --extra dev            # Include development tools
```

### System Dependencies

* **Linux:**
    ```bash
    apt-get install libglib2.0-0 libsm6 libxrender1 libxext6 graphviz
    ```

* **macOS:** See the [Mac installation guide](https://github.com/fastestimator/fastestimator/blob/master/installation_docs/mac_installation.md)

* **Windows:**
    * Install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)
    * Install [Visual C++ redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

* **Optional (for Traceability reports):** LaTeX (`texlive-latex-base`, `texlive-latex-extra`)

### Alternative: pip install

* Stable:
    ```bash
    pip install fastestimator
    ```
    Note: PyTorch is included in the base dependencies. On macOS you get MPS support; on Linux/Windows you get CPU builds.

* Nightly:
    ```bash
    pip install fastestimator-nightly
    ```

* For GPU with CUDA (Linux/Windows only):
    ```bash
    pip install fastestimator
    pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121
    ```

## Docker Hub

Docker containers create isolated virtual environments that share resources with a host machine. Docker provides an easy way to set up a FastEstimator environment. You can simply pull our image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags) and get started:

* Stable:
  * GPU:

      ``` bash
      docker pull fastestimator/fastestimator:latest-gpu
      ```

  * CPU:

      ``` bash
      docker pull fastestimator/fastestimator:latest-cpu
      ```

* Nightly:
  * GPU:

      ``` bash
      docker pull fastestimator/fastestimator:nightly-gpu
      ```

  * CPU:

      ``` bash
      docker pull fastestimator/fastestimator:nightly-cpu
      ```

## Useful Links

* [Website](https://www.fastestimator.org): More info about FastEstimator API and news.
* [Tutorial Series](https://github.com/fastestimator/fastestimator/tree/master/tutorial): Everything you need to know about FastEstimator.
* [Application Hub](https://github.com/fastestimator/fastestimator/tree/master/apphub): End-to-end deep learning examples in FastEstimator.

## Citation

Please cite FastEstimator in your publications if it helps your research:

```
@misc{fastestimator,
  title  = {FastEstimator: A Deep Learning Library for Fast Prototyping and Productization},
  author = {Xiaomeng Dong and Junpyo Hong and Hsi-Ming Chang and Michael Potter and Aritra Chowdhury and
            Purujit Bahl and Vivek Soni and Yun-Chan Tsai and Rajesh Tamada and Gaurav Kumar and Caroline Favart and
            V. Ratna Saripalli and Gopal Avinash},
  note   = {NeurIPS Systems for ML Workshop},
  year   = {2019},
  url    = {http://learningsys.org/neurips19/assets/papers/10_CameraReadySubmission_FastEstimator_final_camera.pdf}
}
```

## License

[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)
