"""
CLiFF-map Python Package Setup

A comprehensive Python implementation of Circular-Linear Flow Field (CLiFF) mapping
for dynamic environment analysis with enhanced features including parallel processing,
checkpointing, and advanced visualization capabilities.
"""

from setuptools import setup, find_packages
import os


# Read README file
def read_readme():
    """Read README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


setup(
    name="cliffmap",
    version="1.0.0",
    author="CLiFF-map Development Team",
    author_email="author@example.com",
    description="Circular-Linear Flow Field mapping for dynamic environment analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cliffmap-python",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/cliffmap-python/issues",
        "Documentation": "https://github.com/yourusername/cliffmap-python/wiki",
        "Source Code": "https://github.com/yourusername/cliffmap-python",
    },
    
    packages=find_packages(),
    package_data={
        'cliffmap': [
            '*.py',
            'examples/*.py',
            'tests/*.py',
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    
    python_requires=">=3.7",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=20.8b1",
            "isort>=5.0",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.7",
            "pandoc>=1.0",
        ],
        "examples": [
            "jupyter>=1.0",
            "notebook>=6.0",
            "ipywidgets>=7.0",
        ],
        "performance": [
            "numba>=0.50",
            "joblib>=1.0",
            "memory_profiler>=0.60",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "cliffmap-analyze=cliffmap.cli:main",
            "cliffmap-benchmark=cliffmap.benchmark:main",
            "cliffmap-visualize=cliffmap.visualization:main",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
    
    keywords=[
        "machine learning",
        "flow field analysis",
        "dynamic mapping",
        "circular statistics",
        "mean shift",
        "expectation maximization",
        "traffic analysis",
        "pedestrian flow",
        "spatial analysis",
        "time series",
        "clustering",
        "unsupervised learning",
    ],
    
    # Package metadata
    platforms=["any"],
    license="MIT",
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
    ],
    
    # Development status
    download_url="https://github.com/yourusername/cliffmap-python/archive/v1.0.0.tar.gz",
)