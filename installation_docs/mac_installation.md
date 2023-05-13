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

## Step 2: Install Miniforge

```bash
CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(brew --prefix zlib)/include -I$(brew --prefix sqlite)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix bzip2)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix sqlite)/lib" pyenv install miniforge3-4.10.3-10
```

```bash
pyenv global miniforge3-4.10.3-10
```

## Step 3: Create a python environment

```bash
conda update --force conda -y
```

```bash
conda create -n femos38 python=3.8
```

```bash
conda activate femos38
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
python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

```bash
conda install jupyter -y
```

```bash
conda install -c conda-forge jupyterlab -y
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
