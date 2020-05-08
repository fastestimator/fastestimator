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
        'albumentations',
        'pyfiglet',
        'opencv-python',
        'scipy',
        'pandas',
        'sklearn',
        'wget',
        'pillow',
        'seaborn',
        'matplotlib',
        'requests',
        'tqdm',
        'h5py',
        'jsonpickle',
        'python-docx',
        'tensorflow_probability==0.8.0',
        'transformers==2.4.1',
        'tensorflow==2.1.0'
    ]
    if os.name == "nt":
        dependencies.append(
            "pycocotools @ git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI")
    else:
        dependencies.append('pycocotools-fix')
        dependencies.append('torch==1.4.0')
        dependencies.append('torchvision==0.5.0')
    return dependencies


setup(
    entry_points={"console_scripts": ["fastestimator = fastestimator.cli.main:run"]},
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
    keywords="fastestimator tensorflow",
    classifiers=["License :: OSI Approved :: Apache Software License", "Programming Language :: Python :: 3"],

    # Declare minimal set for installation
    install_requires=get_dependency(),
    # Declare extra set for installation
    extras_require={})
