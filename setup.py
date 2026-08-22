from setuptools import setup, find_packages

setup(
    name="mot5-ultimate",
    version="2.1.0",  # ✅ New version
    description="MOT5 Ultimate - Sirf 3 Lines! Auto-Sync, Never Forgets!",
    author="Rohit Patra",
    packages=find_packages(),
    install_requires=["numpy>=1.21.0", "requests>=2.25.0"],
    python_requires=">=3.8",
)
