#!/bin/bash

# install.sh

# --
# Setup environment

conda create -y -n mgmmf_env python=3.9
conda activate mgmmf_env

pip install -r requirements.txt
conda install -y -c anaconda cmake=3.22

# --
# Install

./build.sh
pip install -e .