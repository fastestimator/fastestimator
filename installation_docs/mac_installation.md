# MacOS Installation

This guide walks you through installing FastEstimator on macOS using [UV](https://docs.astral.sh/uv/), a fast Python package manager.

## Step 1: Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

## Step 2: Install System Dependencies

Install Homebrew if you don't have it:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install required system packages:
```bash
brew install graphviz
```

**Optional:** For [Traceability report generation](https://github.com/fastestimator/fastestimator/blob/master/tutorial/advanced/t10_report_generation.ipynb), install LaTeX (~5GB):
```bash
brew install --cask mactex
eval "$(/usr/libexec/path_helper)"  # Or restart your terminal
```

## Step 3: Install FastEstimator

### Option A: For Users (pip install)

Create a new project directory and install:
```bash
mkdir my-fe-project && cd my-fe-project
uv init
uv add fastestimator
```

Or install globally:
```bash
uv tool install fastestimator
```

### Option B: For Developers (from source)

Clone the repository:
```bash
git clone https://github.com/fastestimator/fastestimator.git
cd fastestimator
```

Create environment and install dependencies:
```bash
uv sync
```

To also install Jupyter support:
```bash
uv sync --extra jupyter
```

To include development tools (pytest, coverage, linting):
```bash
uv sync --extra dev --extra jupyter
```

## Step 4: Verify Installation

Activate the environment and test:
```bash
# If installed from source
source .venv/bin/activate

# Verify FastEstimator
python -c "import fastestimator as fe; print(fe.__version__)"

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

## GPU Acceleration (Apple Silicon)

On Macs with Apple Silicon (M1/M2/M3/M4), PyTorch supports GPU acceleration via **Metal Performance Shaders (MPS)**. The CPU build of PyTorch includes MPS support automatically.

### Verify MPS Availability

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Using MPS in Your Code

FastEstimator will automatically detect and use MPS when available. You can also explicitly use it:

```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move tensors/models to MPS
tensor = torch.randn(3, 3).to(device)
```

### MPS Limitations

- MPS is supported on macOS 12.3+ with Apple Silicon or AMD GPUs
- Some PyTorch operations may fall back to CPU if not implemented for MPS
- For maximum compatibility, the CPU extra works on all Mac hardware

## UV Virtual Environment Best Practices

### Understanding UV's Virtual Environment

UV automatically creates a `.venv` directory in your project when you run `uv sync`. This is the recommended approach for development.

### Project Structure

```
fastestimator/
├── .venv/              # Virtual environment (auto-created by uv sync)
├── .python-version     # Specifies Python version for UV
├── pyproject.toml      # Project configuration and dependencies
├── uv.lock             # Lock file (auto-generated, commit to git)
└── fastestimator/      # Source code
```

### Development Workflow

```bash
# Initial setup (creates .venv and installs all dependencies)
uv sync --extra dev --extra jupyter

# Activate the environment (optional - uv run works without activation)
source .venv/bin/activate

# Run commands without activation using uv run
uv run python my_script.py
uv run pytest
uv run fastestimator train mnist.py

# Add a new dependency
uv add <package-name>

# Add a dev-only dependency
uv add --dev <package-name>

# Update all dependencies to latest compatible versions
uv sync --upgrade

# Update a specific package
uv add <package-name>@latest

# Remove a dependency
uv remove <package-name>
```

### Lock File Management

UV generates a `uv.lock` file that pins exact versions of all dependencies:

```bash
# Regenerate lock file from pyproject.toml
uv lock

# Install exactly what's in the lock file
uv sync --frozen
```

**Tip:** Commit `uv.lock` to version control for reproducible builds.

### Multiple Python Versions

UV can manage multiple Python versions:

```bash
# Install a specific Python version
uv python install 3.11

# Create environment with specific Python
uv venv --python 3.11

# Sync with specific Python
uv sync --python 3.11
```

### Clean Reinstall

If you need to start fresh:

```bash
# Remove existing environment
rm -rf .venv

# Recreate from lock file
uv sync
```

## Common Commands

```bash
# Run a training script
uv run fastestimator train my_script.py

# Or activate the environment first
source .venv/bin/activate
fastestimator train my_script.py

# Add additional packages
uv add <package-name>

# Update all packages
uv sync --upgrade

# See installed packages
uv pip list

# Export to requirements.txt (for compatibility)
uv pip compile pyproject.toml -o requirements.txt
```

## Troubleshooting

### LaTeX not found for Traceability
Make sure MacTeX is installed and in your PATH:
```bash
which pdflatex
```
If not found, run `eval "$(/usr/libexec/path_helper)"` or restart your terminal.

### OpenCV issues
If you encounter OpenCV-related errors, ensure you have the required system libraries:
```bash
brew install libpng libjpeg
```

### MPS Errors
If you encounter MPS-related errors:

1. Ensure you're on macOS 12.3 or later:
   ```bash
   sw_vers
   ```

2. Try falling back to CPU by setting the environment variable:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

3. Or disable MPS entirely in your code:
   ```python
   import os
   os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
   # Or force CPU
   device = torch.device("cpu")
   ```

### UV Cache Issues
If you encounter strange dependency issues:
```bash
# Clear UV's cache
uv cache clean

# Reinstall
rm -rf .venv
uv sync
```
