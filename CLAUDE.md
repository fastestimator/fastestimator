# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastEstimator is a high-level deep learning library built on PyTorch. It provides a clean, modular API for building and training deep learning models.

**Current Version**: 2.0.0 | **Python**: 3.10-3.12 | **PyTorch**: 2.3.1

## Common Commands

### Environment Setup (UV)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync                        # Base install
uv sync --extra jupyter        # With Jupyter support
uv sync --extra dev            # With dev tools (coverage, yapf, isort)

# Run commands
uv run python script.py
uv run fastestimator train script.py
```

### Running Tests
```bash
# Run all PR tests
cd test && python3 -m unittest discover PR_test

# Or from project root
python3 -m unittest discover test/PR_test

# Run a single test file
python3 -m unittest test/PR_test/unit_test/backend/test_abs.py

# Run a specific test case
python3 -m unittest PR_test.unit_test.backend.test_abs.TestAbs

# Run tests with coverage
cd test
coverage run --source ../fastestimator -m unittest discover PR_test
coverage report  # or: coverage html
```

### CLI Commands
```bash
fastestimator train <script.py>   # Train a model
fastestimator test <script.py>    # Test a model
fastestimator run <script.py>     # Run inference
```

### Code Formatting
```bash
yapf -i <file.py>      # Format a file
isort <file.py>        # Sort imports
```
YAPF config is in `pyproject.toml` under `[tool.yapf]`. Uses PEP8 base style with 120 column limit.

## Architecture

FastEstimator is built on three core pillars:

```
Pipeline → Network → Estimator
   ↓          ↓          ↓
 Data    Computation  Training
```

### Core Classes
- **Pipeline** (`pipeline.py`): Data loading and preprocessing using NumpyOps
- **Network** (`network.py`): Model definition and computation using TensorOps
- **Estimator** (`estimator.py`): Training orchestration combining Pipeline + Network + Traces

### The Operator Pattern
All operations follow a modular pattern with defined inputs/outputs:
- **Op** (`op/op.py`): Base class for all operators
- **NumpyOp** (`op/numpyop/`): CPU operations for data preprocessing
- **TensorOp** (`op/tensorop/`): GPU/CPU tensor operations (models, losses, etc.)
- **Trace** (`trace/`): Training callbacks for monitoring/control

### Key Modules
- `backend/`: Tensor operations (PyTorch-based)
- `architecture/`: Pre-built model architectures
- `dataset/`: Data loading utilities
- `schedule/`: Learning rate and parameter scheduling
- `trace/`: Training callbacks (metrics, logging, model saving)
- `xai/`: Explainable AI utilities

## Test Structure

Tests mirror the source structure:
- `test/PR_test/unit_test/` → `fastestimator/`
- `test/PR_test/integration_test/` → Multi-module tests

Unit tests involve only the tested module. Integration tests involve multiple modules.

The project uses Python's built-in `unittest` framework (not pytest).

## Development Notes

- Mac-specific: tkinter is disabled for multiprocessing compatibility
- OpenCV threads are disabled (`cv2.setNumThreads(0)`) to avoid PyTorch DataLoader conflicts
- Uses `lazy_loader` for efficient module imports
- History logging is auto-disabled during test runs
- GPU support: CUDA on Linux/Windows, MPS on Apple Silicon Macs

## Package Management

The project uses UV for package management with `pyproject.toml`. Key files:
- `pyproject.toml`: Dependencies and UV configuration
- `setup.py`: Legacy pip support (kept for compatibility)
- `.python-version`: Specifies Python 3.10

PyTorch is automatically installed with the correct build:
- macOS: Standard PyTorch with MPS support
- Linux/Windows: CPU-optimized build (GPU users override manually)

## Application Hub

The `apphub/` directory contains end-to-end examples covering:
- Image classification, generation, segmentation
- Object detection, NLP tasks
- Adversarial training, neural architecture search
