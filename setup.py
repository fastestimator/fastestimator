import datetime
import os
import re
import sys

from setuptools import find_packages, setup

is_nightly = os.environ.get('FASTESTIMATOR_IS_NIGHTLY', None)
if is_nightly is not None:
    sys.stderr.write("Using '%s=%s' environment variable!\n" % ('FASTESTIMATOR_IS_NIGHTLY', is_nightly))


def get_version():
    path = os.path.dirname(__file__)
    version_re = re.compile(r'''__version__ = ['"](.+)['"]''')
    with open(os.path.join(path, 'fastestimator', '__init__.py')) as f:
        init = f.read()

    now = datetime.datetime.now()
    version = version_re.search(init).group(1)
    if is_nightly:
        return "{}.dev{:04}{:02}{:02}{:02}{:02}".format(version, now.year, now.month, now.day, now.hour, now.minute)
    else:
        return version


def get_name():
    if is_nightly:
        return "fastestimator-nightly"
    else:
        return "fastestimator"


def get_dependency():
    dependencies = [
        'albumentations==1.3.1',
        'matplotlib==3.7.1',
        'h5py==3.8.0',
        'scipy==1.9.1',
        'PyLaTeX==1.4.1',
        'natsort==8.3.1',
        'tensorflow_probability==0.19.0',
        'tensorflow-addons==0.19.0',
        'transformers==4.26.1',
        'torchinfo==1.7.2',
        'graphviz==0.20.1',
        'torchview==0.2.6',
        'pydot==1.4.2',
        'dot2tex==2.11.3',
        'gdown==4.6.4',
        'PySocks==1.7.1',
        'uncertainty-calibration==0.1.4',
        'dill==0.3.6',
        'scikit-image==0.20.0',
        'prettytable==3.6.0',
        'nltk==3.8.1',
        'requests==2.28.2',
        'tqdm==4.65.0',
        'numpy==1.24.2',
        'pyfiglet==0.8.post1',
        'opencv-python==4.7.0.72',
        'pandas==2.0.1',
        'wget==3.2',
        'pillow==9.4.0',  # See discussion below before changing this in the future
        'jsonpickle==3.0.1',
        'python-docx==0.8.11',
        'plotly==5.13.1',
        'kaleido==0.2.1',
        'orjson==3.8.7',
        'scikit-learn==1.2.2',
        'lazy_loader==0.1',
        'fe_pycocotools==1.0',
        'typing_extensions==4.5.0',
        'charset-normalizer==3.1.0',
        'py-cpuinfo==9.0.0'
    ]
    return dependencies


setup(
    entry_points={"console_scripts": ["fastestimator = fastestimator.cli.main:main"]},
    name=get_name(),
    version=get_version(),
    description="Deep learning framework",
    packages=find_packages(),
    package_dir={'': '.'},
    long_description="FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily \
                    build a high-performance deep learning model and run it anywhere.",
    author="FastEstimator Dev",
    url='https://github.com/fastestimator/fastestimator',
    license="Apache License 2.0",
    keywords="fastestimator tensorflow pytorch",
    classifiers=["License :: OSI Approved :: Apache Software License", "Programming Language :: Python :: 3"],
    # Declare minimal set for installation
    install_requires=get_dependency(),
    python_requires='>=3.8.0',
    # Declare extra set for installation
    extras_require={})

# Pillow version warning: Version < 9.5.0 broke the CPU docker image on dnn.ipynb and yolov5.ipynb, causing a Kernel
# died error and segmentation fault respectively. This was fixed by importing the non-fe dependencies after importing
# FE related dependencies. Version 9.5.0 fixes those problems but causes dnn_tf.py, alocc_tf.py, yolov5_tf.py, and
# retinanet_tf.py all to crash (again on the CPU-only docker image), all with segmentation faults. Strangely, the more
# direct cause of the crash in those cases is that if you import sklearn (or albumentations which in turn imports
# sklearn) before importing tensorflow then you will get a segfault during tensorflow training. You can solve this by
# importing torch before sklearn. You can also import torch after importing sklearn to fix the problem, but if someone
# imports sklearn AND cv2 AND tensorflow, then importing torch later can't salvage the situation. If we could ensure
# that torch was imported first then this wouldn't be an issue, but I don't see a way to do that at the moment. In
# another bizarre twist, skl -> PIL -> tf -> torch does not result in a segfault. It's not clear at the moment why
# changing the PIL version helps whereas changing the skl or cv2 versions does not, given that PIL importing doesn't
# cause a direct problem. This note has been included in order to save a week of debugging time next time we do version
# bumps. If you want a more recent version of PIL, and are on a non-linux machine or have GPUs, then feel free to
# manually install a higher version after installing FE and just ignore the pip incompatibility warning.
