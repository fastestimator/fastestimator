# Changelog

## FastEstimator 2.0.0

### Highlights

FastEstimator 2.0 is a major release that modernizes the framework with a focus on simplicity, performance, and developer experience.

### Breaking Changes

* **TensorFlow removed.** FastEstimator is now a **PyTorch-only** framework. All TensorFlow backends, architectures, layers, ops, and application hub examples have been removed. Users migrating from v1.x should convert any TensorFlow models and code to PyTorch equivalents.
* **Python 3.10+** is now the minimum supported version (previously 3.6+).
* **UV-based dependency management.** The project now uses [UV](https://docs.astral.sh/uv/) and `pyproject.toml` as the primary build/dependency configuration. `setup.py` is retained for backward compatibility.

### New Features

* **JAX backend support.** Added `jax` and `jaxlib` as dependencies for JAX-based computation.
* **Modernized build system.** Migrated to `pyproject.toml` with [Hatchling](https://hatch.pypa.io/) as the build backend and UV for fast, reproducible installs.
* **Optional dependency groups.** Install extras with `uv sync --extra jupyter` or `uv sync --extra dev` for Jupyter and development tools respectively.
* **Platform-aware PyTorch.** PyTorch is bundled as a direct dependency with platform-specific index sources — macOS gets MPS (Apple Silicon GPU) support, Linux/Windows get CPU-optimized builds by default.

### Improvements

* **Backend functions.** All backend functions (`abs`, `argmax`, `cast`, `concat`, `crossentropy`, `focal_loss`, `gather`, `gradient`, `reduce_*`, `reshape`, `transpose`, etc.) have been streamlined to PyTorch-only implementations.
* **Ops (NumpyOp & TensorOp).** All numpy and tensor operations have been updated and simplified, removing TensorFlow code paths.
* **Traces.** All traces (metrics, IO, adapt, XAI) updated for PyTorch-only execution.
* **Datasets.** Dataset modules updated with improved data loading.
* **Architectures.** TensorFlow architecture implementations removed; PyTorch architectures (`LeNet`, `ResNet9`, `UNet`, `AttentionUNet`, `WideResNet`) retained and updated.
* **Layers.** TensorFlow layer implementations removed; PyTorch layers retained.
* **Schedules.** Learning rate schedules and schedule utilities updated.
* **Search.** Hyperparameter search modules (grid search, golden section) updated.
* **Slicers.** Slicer modules updated for PyTorch compatibility.
* **Summary & System.** Summary and system tracking modules updated.
* **CLI.** Command-line tools (`train`, `logs`, `plot`) updated.
* **Tutorials.** All beginner, advanced, and XAI tutorials updated for PyTorch-only usage.
* **Application Hub.** All TensorFlow example scripts removed. PyTorch examples retained and updated. Application Hub README updated to reflect PyTorch-only examples.
* **Docker images.** Dockerfiles for both CPU and GPU updated for the new dependency stack.
* **macOS installation guide.** Updated to use UV-based workflow with MPS GPU acceleration instructions.

### Dependency Updates

| Package | Version |
|---------|---------|
| PyTorch | 2.3.1 |
| torchvision | 0.18.1 |
| torchaudio | 2.3.1 |
| CUDA (GPU builds) | 12.2.2 |
| NumPy | 1.26.4 |
| JAX | 0.4.33 |
| transformers | 4.38.2 |
| albumentations | 1.4.1 |
| scikit-learn | 1.3.2 |
| scipy | latest |
| matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| pandas | 2.0.1 |

### Removed

* `fastestimator.architecture.tensorflow` — all TensorFlow model architectures
* `fastestimator.layers.tensorflow` — all TensorFlow custom layers
* All `*_tf.py` application hub examples
* `installation_docs/tensorflow_windows_installation.md`
* TensorFlow code paths from all backend functions, ops, and traces

### Migration Guide

If upgrading from FastEstimator 1.x:

1. **Replace TensorFlow models** with PyTorch equivalents. FastEstimator provides built-in PyTorch architectures in `fastestimator.architecture.pytorch`.
2. **Update imports.** Remove any `tensorflow` or `_tf` references from your code.
3. **Use `pyproject.toml`** for dependency management. Run `uv sync` to install all dependencies.
4. **GPU setup.** For CUDA GPU support, install the CUDA-enabled PyTorch after installing FastEstimator:
   ```bash
   pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121
   ```

## Previous Releases

* [FastEstimator 1.7](https://github.com/fastestimator/fastestimator/tree/r1.7)
* [FastEstimator 1.6](https://github.com/fastestimator/fastestimator/tree/r1.6)
* [FastEstimator 1.5](https://github.com/fastestimator/fastestimator/tree/r1.5)
* [FastEstimator 1.4](https://github.com/fastestimator/fastestimator/tree/r1.4)
* [FastEstimator 1.3](https://github.com/fastestimator/fastestimator/tree/r1.3)
* [FastEstimator 1.2](https://github.com/fastestimator/fastestimator/tree/r1.2)
* [FastEstimator 1.1](https://github.com/fastestimator/fastestimator/tree/r1.1)
