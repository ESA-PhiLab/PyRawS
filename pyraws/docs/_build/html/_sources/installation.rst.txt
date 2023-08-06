Installation
=================

PyRawS is a Python package that can be installed in different ways. The installation process depends on the OS you are using. We suggest to use Linux or Mac OS.
You can install PyRawS in the following ways:

**Table of contents**

.. contents::
   :local:
   :depth: 1
   

a) Via pip
----------

You can install pyraws from pypi, from CLI execute:

You can install  `pyraws` with [pypy](https://www.pypy.org/) by running the following command from CLI:

.. code-block:: bash

   pip install pyraws


b) Build from Source
--------------------

Before all, clone this repository. We suggest using git from CLI, execute:

.. code-block:: bash

   git clone https://github.com/ESA-PhiLab/PyRawS


Create the PyRawS environment


- On Linux

  .. code-block:: bash

     # For Linux, the installation is straightforward. 
     # You just need to run the following command from the main directory:
     \bin\bash\ source pyraws_install.sh

     # NB: Your data should be placed in the data directory in the main.

- On Other OS

  .. code-block:: bash

     # To install the environment, we suggest to use anaconda. 
     # You can create a dedicated conda environment by using the `environment.yml` file by running the following command from the main directory: 
     conda env create -f environment.yml 

     # To activate your environment, please execute:
     conda activate PyRawS






c) With Docker
--------------

To use PyRawS with docker, use one of the following methods.

*Method 1: Pull the docker image from Docker Hub*

.. code-block:: bash

   docker pull sirbastiano94/pyraws:latest

*Method 2: Build docker image*

Follow these steps:

1. Clone the repository and build the docker image by running the following command from the main directory:

   .. code-block:: bash

      docker build -t pyraws:latest  --build-arg CACHEBUST=$(date +%s) -f dockerfile .

2. Run the docker image by executing:

   .. code-block:: bash

      docker run -it --rm -p 8888:8888 pyraws:latest
