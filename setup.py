import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="pyrack",
    version="0.0.1",
    description="A python package to analyze different types of documents",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/realpython/reader",
    author="Siddharth Venkateswaran, Kumar Sundaram, Aravindhan Dhanaraj, Sreenath Gopi",
    author_email="kumarsund3@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["pyrack"],
    include_package_data=True,
    install_requires=["pandas", "numpy", "matplotlib", "imageai", "opencv-python>=3.4.6.27"],
)