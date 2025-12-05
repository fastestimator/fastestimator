# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastEstimator is a high-level deep learning library built on TensorFlow 2 and PyTorch. It provides a unified API that works seamlessly with both frameworks.

**Current Version**: 1.7.0 | **Python**: 3.10-3.12 | **TensorFlow**: 2.15.1 | **PyTorch**: 2.3.1

## Common Commands

### Running Tests
```bash
# Run all PR tests
python3 -m unittest discover test/PR_test

# Run a single test file
python3 -m unittest test/PR_test/unit_test/backend/test_abs.py

# Run a specific test case
python3 -m unittest PR_test.unit_test.backend.test_abs.TestAbs

# Run tests with coverage
coverage run --source ../fastestimator -m unittest discover test/PR_test
coverage report  # or: coverage html
```

### CLI Commands
```bash
fastestimator train <script.py>   # Train a model
fastestimator test <script.py>    # Test a model
fastestimator run <script.py>     # Run inference
```

### Code Formatting
YAPF formatter with PEP8 base style, 120 column limit. Config in `.style.yapf`.

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
- `backend/`: Framework-agnostic tensor operations (68 modules abstracting TF/PyTorch differences)
- `architecture/`: Pre-built model architectures for both TensorFlow and PyTorch
- `dataset/`: Data loading utilities
- `schedule/`: Learning rate and parameter scheduling
- `trace/`: Training callbacks (metrics, logging, model saving)
- `xai/`: Explainable AI utilities

### Framework Detection
FastEstimator auto-detects whether models/tensors are TensorFlow or PyTorch and routes operations accordingly through the `backend/` module.

## Test Structure

Tests mirror the source structure:
- `test/PR_test/unit_test/` → `fastestimator/`
- `test/PR_test/integration_test/` → Multi-module tests

Unit tests involve only the tested module. Integration tests involve multiple modules.

## Development Notes

- Mac-specific: tkinter is disabled for multiprocessing compatibility
- OpenCV threads are disabled (`cv2.setNumThreads(0)`) to avoid PyTorch DataLoader conflicts
- Uses `lazy_loader` for efficient module imports
- History logging is auto-disabled during test runs

## Application Hub

The `apphub/` directory contains end-to-end examples covering:
- Image classification, generation, segmentation
- Object detection, NLP tasks
- Adversarial training, neural architecture search
- Each example has both TensorFlow and PyTorch implementations
