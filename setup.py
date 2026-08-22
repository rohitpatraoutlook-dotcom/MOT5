from setuptools import setup, find_packages

setup(
    name="mot5",
    version="4.0.0",
    description="MOT5 - Equation Discovery Engine with Auto-Sync",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rohit Patra",
    author_email="rohitpatraoutlook@gmail.com",
    url="https://github.com/rohitpatraoutlook-dotcom/MOT5",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "requests>=2.25.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
