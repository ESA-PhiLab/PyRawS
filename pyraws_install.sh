#!/bin/bash

# This script installs the pyraws package and its dependencies
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment called pyraws
conda create --name pyraws python=3.9 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate pyraws; then
    echo "pyraws environment activated"
else
    echo "pyraws environment not found"
    exit 1
fi


# Define PyTorch version and components
pytorch_version="1.11.0"
torchvision_version="0.12.0"
torchaudio_version="0.11.0"

# Function to install PyTorch
install_pytorch() {
  conda install -y pytorch==$pytorch_version torchvision==$torchvision_version torchaudio==$torchaudio_version $1 -c pytorch
}

# Install via pip:
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Detect operating system
case "$OSTYPE" in
  "darwin"*)
    # macOS
    echo "Detected macOS"
    install_pytorch "cpuonly"
    ;;
  "linux-gnu"|"linux-gnueabihf")
    # Linux
    echo "Detected Linux"
    # Install PyTorch with CUDA 11.3 support
    install_pytorch "cudatoolkit=11.3"
    ;;
  "msys"|"win32")
    # Windows
    echo "Detected Windows"
    # Install PyTorch with CUDA 11.3 support
    install_pytorch "cudatoolkit=11.3"
    ;;
  *)
    echo "Unsupported operating system"
    exit 1
    ;;
esac


# get absolute path of current working directory
# and setup the sys_cfg.py file
echo "PYRAWS_HOME_PATH = '$(pwd)'" > pyraws/sys_cfg.py
echo "DATA_PATH = '$(pwd)/data'" >> pyraws/sys_cfg.py

# install pyraws
pip install -e .
