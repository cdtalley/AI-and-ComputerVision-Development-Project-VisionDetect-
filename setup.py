from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="visiondetect",
    version="0.1.0",
    author="AI Developer",
    author_email="user@example.com",
    description="A comprehensive computer vision framework for object detection using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/visiondetect",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.2.0",
        "tqdm>=4.60.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "albumentations>=1.0.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
    ],
)
