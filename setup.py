# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from setuptools import setup, find_packages

# Function to parse requirements.txt
def load_requirements(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        # Filter out empty lines and comments
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="defm",
    version="0.1.0",
    author="Manthan Patel",
    description="DeFM: Learning Foundation Representations from Depth for Robotics",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["defm", "defm.*"]),
    python_requires=">=3.10",
    
    # --- Link to requirements.txt ---
    install_requires=load_requirements("requirements.txt"),
    
    include_package_data=True,
    package_data={
        "defm": ["configs/*.yaml"],
    },
)