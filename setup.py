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
        'numpy',
        'albumentations==0.5.2',
        'pyfiglet',
        'opencv-python',
        'pandas',
        'sklearn',
        'wget',
        'pillow',
        'seaborn',
        'matplotlib==3.3.3',
        'requests',
        'tqdm',
        'h5py==2.10.0',
        'jsonpickle',
        'python-docx',
        'scipy==1.5.4',
        'PyLaTeX==1.4.1',
        'natsort==7.1.0',
        'tensorflow_probability==0.12.1',
        'tensorflow-addons==0.12.1',
        'transformers==4.6.0',
        'pytorch-model-summary==0.1.2',
        'graphviz==0.15',
        'hiddenlayer==0.3',
        'pydot==1.4.1',
        'dot2tex==2.11.3',
        'gdown==3.12.0',
        'PySocks==1.7.1',
        'uncertainty-calibration==0.0.8',
        'dill==0.3.3'
    ]
    if os.name == "nt":
        dependencies.append(
            "pycocotools @ git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI")
    else:
        dependencies.append('pycocotools-fix')
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
    # Declare extra set for installation
    extras_require={})
