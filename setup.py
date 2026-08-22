from setuptools import setup, find_packages

setup(
    name="mot5",
    version="11.0.0",
    description="MOT5 - Smart Equation Discovery Engine",
    long_description="Discover equations from data in 10 lines! No random search, no forgetting!",
    author="MOT5 Team",
    packages=find_packages(),
    install_requires=["numpy>=1.21.0"],
    python_requires=">=3.8",
)
