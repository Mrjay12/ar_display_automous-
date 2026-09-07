"""Setup script for GPS-Denied Visual Localization System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="ar-display-autonomous",
    version="0.1.0",
    description="GPS-Denied Visual Localization & AR Mapping System using OAK-D Pro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Visual Localization Team",
    license="TBD",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "depthai>=2.24.0",
        "opencv-contrib-python>=4.8.0",
        "pyproj>=3.6.0",
        "geographiclib>=1.52",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0",
        "imageio>=2.32.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "h5py>=3.9.0",
        "scikit-image>=0.21.0",
        "open3d>=0.17.0",
        "trimesh>=3.21.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future: CLI entry points
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
