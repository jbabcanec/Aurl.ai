"""
Setup configuration for Aurl.ai music generation system.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from src/__init__.py
def get_version():
    version = {}
    with open("src/__init__.py", "r", encoding="utf-8") as fh:
        exec(fh.read(), version)
    return version["__version__"]

setup(
    name="aurl",
    version=get_version(),
    author="Aurl.ai Team",
    author_email="contact@aurl.ai",
    description="State-of-the-art music generation AI system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/aurl-ai/aurl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.1",
            "sox>=1.4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "aurl-train=train_pipeline:main",
            "aurl-generate=generate_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aurl": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "music generation",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "transformer",
        "VAE",
        "GAN",
        "MIDI",
        "neural networks",
        "music AI",
    ],
    project_urls={
        "Bug Reports": "https://github.com/aurl-ai/aurl/issues",
        "Source": "https://github.com/aurl-ai/aurl",
        "Documentation": "https://docs.aurl.ai/",
    },
)