from __future__ import absolute_import


from setuptools import setup, find_packages

setup(name="fastestimator",
      version="2.0",
      description="Deep learning Application framework",
      packages=find_packages(),
      package_dir={'': '.'},
      long_description="FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere.",
      author="FastEstimator Dev",
      url='https://github.build.ge.com/EdisonAITK/FastEstimator',
      license="Apache License 2.0",
      keywords="keras tensorflow",
      classifiers=[
          "Development Status :: 3 - Development/Stable",
          "Intended Audience :: Developers",
          "Natural Language :: English",
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.5",
      ],

      # Declare minimal set for installation
      install_requires=[
          'numpy',
          'pyfiglet',
          'pandas',
          'pillow',
          'sklearn',
          'wget',
          'matplotlib',
          'seaborn>= 0.9.0',
          'scipy',
          'pytest',
          'pytest-cov',
          'tensorflow-addons',
          'tensorflow-probability',
          'umap-learn',
          'tqdm'
      ],
      # Declare extra set for installation
      extras_require={
      },
      scripts=['bin/fastestimator']
      )
