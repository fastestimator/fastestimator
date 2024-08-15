# MacOS Installation

This is a guide for how to install FastEstimator on MacOS starting from a brand new machine. If you already have some of the dependencies that are mentioned here, you can skip down to where the steps start becoming relevant for you.

## Step 0: Install Homebrew

If you don't have brew installed yet, you can install it following the instructions [here](https://docs.brew.sh/Installation).

## Step 1: Install System Dependencies

```bash
brew update && brew install pyenv pyenv-virtualenv
```

```bash
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
```

```bash
brew install zlib sqlite bzip2 libiconv libzip openssl rust
```

The following graphviz and mactex dependencies are kind of large (~5GB), so you can skip them if you don't want to use [auto-doc generation features](https://github.com/fastestimator/fastestimator/blob/master/tutorial/advanced/t10_report_generation.ipynb) like [Traceability reports](https://github.com/fastestimator/fastestimator/blob/master/tutorial/resources/t10a_traceability.pdf). You can also choose to install them later if you change your mind.

```bash
brew install graphviz
brew install --cask mactex  # You could alternatively get this from https://www.tug.org/mactex/mactex-download.html
eval "$(/usr/libexec/path_helper)"  # You can skip this line if you instead restart your terminal window
```

## Step 2: Install Miniforge

```bash
CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(brew --prefix zlib)/include -I$(brew --prefix sqlite)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix bzip2)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix sqlite)/lib" pyenv install miniforge3-4.14.0-2
```

```bash
pyenv global miniforge3-4.14.0-2
```

## Step 3: Create a python environment

```bash
conda update --force conda -y
```

```bash
conda create -n FE16 python=3.8 -y
```

Use pyenv rather than conda to activate your virtual environment to ensure that both pip and conda will point to the same python environment:

```bash
pyenv activate FE16
```

## Step 4: Install python dependencies

```bash
conda install -c apple tensorflow-deps==2.10.0 -y
```

```bash
python -m pip install tensorflow-macos==2.11.0
```

```bash
python -m pip install tensorflow-metal==0.7.1
```

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

```bash
conda install qtconsole -y
```

```bash
conda install jupyter -y
```

```bash
conda install -c conda-forge jupyterlab -y
```

Jupyterlab smuggles in a different version of urllib3 which we need to get rid of to prevent some warning messages later:

```bash
pip uninstall -y urllib3 && pip install urllib3==1.26.16
```

## Step 5: Install FastEstimator

For regular usage:

```bash
pip install fastestimator
```

For FE developers:

```bash
git clone https://github.com/fastestimator/fastestimator.git
```

```bash
cd fastestimator
```

```bash
pip install -e .
```
