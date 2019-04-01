import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
	
	
setuptools.setup(
    name="fastestimator",
    version="0.0.1",
    author="Edittool",
    author_email="shawnmengdong@gmail.com",
    description="building deep learning models fast&easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fastestimator/fastestimator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)