from setuptools import setup, find_packages

setup(
    name="pg_amf_bearing",
    version="1.0.0",
    description=(
        "Physics-Guided Adaptive Moment Features (PG-AMF) for bearing "
        "fault diagnosis — XJTU Gearbox Dataset"
    ),
    author="",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
    },
)
