#!/bin/bash

# This script installs the pyraws package and its dependencies
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment called openmmlab
conda create --name pyraws python=3.9 -y

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate pyraws; then
    echo "pyraws environment activated"
else
    echo "pyraws environment not found"
    exit 1
fi

#!/bin/bash
# Check the operating system and install pytorch accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  echo "Detected macOS"
  conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
  # Linux
  echo "Detected Linux"
  if command -v nvcc >/dev/null 2>&1; then
    # CUDA is available
    if nvcc --version | grep "release 10\.2" >/dev/null 2>&1; then
      # CUDA 10.2
      echo "Detected CUDA 10.2"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
    elif nvcc --version | grep "release 11\.3" >/dev/null 2>&1; then
      # CUDA 11.3
      echo "Detected CUDA 11.3"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    else
      # CUDA version not supported, installing CPU version
      echo "CUDA version not supported, installing CPU version"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
    fi
  else
    # CUDA is not available, installing CPU version
    echo "CUDA not detected, installing CPU version"
    conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
  fi
elif [[ "$OSTYPE" == "msys" ]]; then
  # Windows
  echo "Detected Windows"
  if where nvcc >/dev/null 2>&1; then
    # CUDA is available
    if nvcc --version | grep "release 10\.2" >/dev/null 2>&1; then
      # CUDA 10.2
      echo "Detected CUDA 10.2"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
    elif nvcc --version | grep "release 11\.3" >/dev/null 2>&1; then
      # CUDA 11.3
      echo "Detected CUDA 11.3"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    else
      # CUDA version not supported, installing CPU version
      echo "CUDA version not supported, installing CPU version"
      conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
    fi
  else
    # CUDA is not available, installing CPU version
    echo "CUDA not detected, installing CPU version"
    conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch 
  fi
else
  echo "Unsupported operating system"
  exit 1
fi

# get absolute path of current working directory
# and setup the sys_cfg.py file
echo "PYRAWS_HOME_PATH = '$(pwd)'" > pyraws/sys_cfg.py
echo "DATA_PATH = '$(pwd)/data'" >> pyraws/sys_cfg.py

# install pyraws
pip install -e .
