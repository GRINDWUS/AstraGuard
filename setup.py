import os
import sys
import setuptools

setuptools.setup(
    name="astraguard-sdk",
    version="2.0.0",
    author="AstraGuard Engineering Team - SIH 2026",
    description="Python SDK for ISRO ATE Hardware Ingestion & AstraGuard Real-Time Reliability API",
    py_modules=["astraguard_sdk"],
    install_requires=[
        "requests>=2.25.0",
        "pandas>=1.2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
