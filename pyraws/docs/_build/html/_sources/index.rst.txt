.. pyraws documentation master file, created by
   sphinx-quickstart on Sun Aug  6 16:01:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyraws's documentation!
==================================

Introduction
============

.. image:: https://img.shields.io/docker/automated/sirbastiano94/pyraws?color=blue
    :target: https://hub.docker.com/repository/docker/sirbastiano94/pyraws/general

.. image:: https://img.shields.io/github/last-commit/ESA-PhiLab/PyRawS?style=flat-square
.. image:: https://img.shields.io/github/contributors/ESA-PhiLab/PyRawS?style=flat-square
.. image:: https://img.shields.io/github/issues/ESA-PhiLab/PyRawS?style=flat-square
.. image:: https://img.shields.io/github/issues-pr/ESA-PhiLab/PyRawS?style=flat-square

.. image:: https://github.com/ESA-PhiLab/PyRawS/actions/workflows/run_tests.yml/badge.svg
    :target: https://github.com/ESA-PhiLab/PyRawS/actions/workflows/run_tests.yml

.. image:: https://img.shields.io/badge/python-3.8-blue.svg
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
.. image:: https://img.shields.io/badge/python-3.10-blue.svg

.. image:: https://img.shields.io/pypi/v/pyraws.svg
    :target: https://pypi.org/project/pyraws/


.. image:: https://i.postimg.cc/7Yptgf5y/Py-Raw-S-logo.png
    :target: https://postimg.cc/vctvy87P

About the project
-----------------

**Python for RAW Sentinel2 data (PyRawS)** is a powerful open-source Python package that provides a comprehensive set of tools for working with Sentinel-2 Raw data. 

.. admonition:: Defining 'Raw Data' in this Project
   :class: note

   In the context of this project, we refer to 'Raw Data' as Sentinel-2 data that is generated through the decompression and the addition of metadata to Sentinel-2 L0 data. In simpler terms, the term 'Raw' is used to denote products that have been decompressed, with relevant ancillary information appended. 

   For a more in-depth understanding, please refer to our `research paper <https://arxiv.org/abs/2305.11891>`_.


.. warning:: This project is currently under development.

License
-----------------

Distributed under the Apache License.


.. toctree::
   :maxdepth: 1
   :caption: First Steps:
   
   installation
   configuration
   contributing
   quickstart
   API

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   database
   raw
   l1
   utils

.. toctree::
   :maxdepth: 1
   :caption: Useful Info:

   glossario
   changelog
   contacts



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
