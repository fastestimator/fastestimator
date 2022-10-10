# Tensorflow Windows Installation

First, go to this [link](https://www.tensorflow.org/install/source_windows#gpu) and check out the python version necessary for TensorFlow. Make sure you have a correct Python version.

## Step 1 : Finding appropriate cuda version

Depending on your TensorFlow version, you will need different CUDA version. For example, in Tensorflow 2.9, you need to get CUDA version 11.2 and cuDNN version 8.1.

## Step 2 : Installing Microsoft Visual Studio

Check the `Compiler` column of your TensorFlow version, `MSVC 2019` means Microsoft Visual Studios (2019). In order to properly compile CUDA, we need to download the corresponding Visual Studio version. For example, for CUDA 11.2 we need to install [Visual Studio 2019](https://docs.microsoft.com/en-us/visualstudio/releases/2019/release-notes).

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/VS.PNG?raw=true" title="Nvidia likes to hide stuff" width=600 height=350>
</p>

Run the downloadable executable to install the visual studio. The installation process will ask to choose what workload to install, for this installation we dont require any workload, hence press continue. Once installation is complete, Visual Studio would want you to sign in, but lets ignore them.

[Reference](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html).

## Step 3 : Installing Nvidia CUDA Toolkit

Finally we are at a point where we can start with geting CUDA to work, but before that let's check if we already have an existing CUDA toolkit and if the existing CUDA is compatible with the required Tensorflow version. Go to Windows setting and choose "Apps and Features" and search for "NVIDIA", you will see something like this.

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/SearchNvidia.PNG?raw=true" title="Lets see if you are ready" width=350 height=600>
</p>

For example, to use Tensorflow 2.9.0, we need NVIDIA CUDA Toolkit version 11.2. From the figure above it can be seen that CUDA Toolkit 11.2 is already installed, but in case your drivers are of any other version we might need to get rid of them. Uninstall all the drivers with "NVIDIA CUDA" in their title and please do NOT touch anything else. Then go to folder "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" and delete the folder with CUDA version as its name.

If in your search for "NVIDIA" in "Apps and Features" and you find that you dont have any CUDA Toolkit installed, go this this [page](https://developer.nvidia.com/cuda-toolkit-archive) which looks like following.

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/CUDA_toolkit.PNG?raw=true" title="Finally we are here" width=450 height=500>
</p>

Here we see that there are three different versions of 11.2 (You can choose any one of them). Click on the version you want, then choose Windows 10 followed by network installer and click download.

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/CUDA_dwnld.PNG?raw=true" title="Finally we are here" width=550 height=500>
</p>

Run the downloaded executable and follow on screen prompts with the default configurations. After this its recommended to restart the computer.

## Step 4 : Install cuDNN version 8.1

Next is to install cuDNN, follow this [link](https://developer.nvidia.com/cudnn) and press download. (As shown below)

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/cuDNN.PNG?raw=true" title="Finally we are here" width=700 height=150>
</p>

To install cuDNN you need a NVIDIA Developer account. So if you already have an account login or else create a new account. Then go to the [cuDNN archive download page](https://developer.nvidia.com/rdp/cudnn-archive) and search for the version you need and install the driver for Windows as shown below.

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/cuDNN_8_1.PNG?raw=true" title="Finally we are here" width=700 height=150>
</p>

This will install a zip folder. Extract the content of the zip folder and migrate to the cuda folder inside the extracted folder. Copy all the content of the folder and paste it at "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2". (If they already exist you should replace the files at the destination)

## Step 5 : Add CUDA toolkit to the PATH:

Now let's make Windows know of CUDA's location by adding few folders to environment variables. In the destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2", there exist a "bin" folder. Copy the path to this folder.

In the windows search bar type "Environment Variables"

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/EV.PNG?raw=true" title="Lets save our Environment" width=500 height=450>
</p>

Clicking on the first result would lead to opening system properties, where we need to select "Environment Variables..."

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/System_prop.PNG?raw=true" title="Lets save our Environment" width=500 height=450>
</p>

Once you do that, you will see a pop up similar to following. In the "User variables", choose "Path" and click edit. Click "New" and paste the path we copied earlier.

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/EV_path.PNG?raw=true" title="Lets save our Environment" width=500 height=450>
</p>

Now let's go back to the destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", there exist a "libnvvp" folder. Copy the path to this folder and follow the same steps as above to add this path to environment variables.

## Step 6 : Install Tensorflow

Now finally we are ready to install tensorflow. A complete guide for tensorflow installation can be found [here](https://www.tensorflow.org/install). Here for simplicity we perform a system install using pip.

Open Powershell or Command Prompt and type :
 `pip3 install --user --upgrade tensorflow`

<p align="center">
  <img src="https://github.com/fastestimator-util/fastestimator-misc/blob/master/resource/pictures/installation_docs/WndPS.PNG?raw=true" title="Lets save our Environment" width=500 height=100>
</p>

Lets ensure that our tensorflow is able to detect the GPU. In your Powershell or Command Prompt type :
`python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

If the output of above mentioned command is something like `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` then congratulations, now you know the secret of installing TensorFlow on Windows.

