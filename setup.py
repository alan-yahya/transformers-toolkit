from setuptools import setup, find_packages
import os

# Read the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transformer_toolkit",
    version="0.1.0",
    packages=find_packages(include=['transformer_toolkit', 'transformer_toolkit.*']),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for working with transformer models and visualizing attention patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/transformer_toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        'transformer_toolkit': ['examples/*.py'],
    },
) 