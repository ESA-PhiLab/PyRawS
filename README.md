[![Docker Automated build](https://img.shields.io/docker/automated/sirbastiano94/pyraws?color=blue)](https://hub.docker.com/repository/docker/sirbastiano94/pyraws/general)
![GitHub last commit](https://img.shields.io/github/last-commit/ESA-PhiLab/PyRawS?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/ESA-PhiLab/PyRawS?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/ESA-PhiLab/PyRawS?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ESA-PhiLab/PyRawS?style=flat-square)
[![Tests](https://github.com/ESA-PhiLab/PyRawS/actions/workflows/run_tests.yml/badge.svg)](https://github.com/ESA-PhiLab/PyRawS/actions/workflows/run_tests.yml)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/pyraws.svg)](https://pypi.org/project/pyraws/1.0.0/)
# PyRawS
[![Py-Raw-S-logo.png](https://i.postimg.cc/7Yptgf5y/Py-Raw-S-logo.png)](https://postimg.cc/vctvy87P)
## About the project
`Python for RAW Sentinel2 data (PyRawS)` is a powerful open-source Python package that provides a comprehensive set of tools for working with [Sentinel-2 Raw data](#sentinel-2-raw-data).
<sup id="fnref:1"><a href="#fn:1" class="footnote">1</a></sup>
It provides utilities for coarse spatial bands coregistration, geo-referencing, data visualization, and image processing.
The software is demonstrated on the first Sentinel-2 Raw database for warm temperature hotspots detection/classification, making it an ideal tool for a wide range of applications in remote sensing and earth observation.
The package is written in Python and is open source, making it easy to use and modify for your specific needs.
The systme is based on [pytorch]("https://pytorch.org/"), which be installed with `CUDA 11` support, to enable GPU acceleation.

TEST 1

NB: What we call raw data in this project are Sentinel-2 data generated by the decompression and metadata addition of Sentinel-2 L0 data. Because of that, with the exception of the effects due to onboard equalization and lossy compression, they are the most similar version of the rawest form of data acquired by the satellite's sensors. Both the compression and equalization are applied onboard the satellite to reduce the amount of data transmitted to the ground station. For easy naming convention, this repo refer to the term "Raw" as the products decompressed with ancillary information appended. For further information browse our paper at https://arxiv.org/abs/2305.11891 <a href="#fnref:1" class="reversefootnote">&#8617;</a></li>

*(Disclaimer: This project is currently under development.)*



## Contributing
The ```PyRawS``` project is open to contributions. To discuss new ideas and applications, please, reach us via email (please, refer to [Contacts](#contacts)). To report a bug or request a new feature, please, open an [issue](https://github.com/ESA-PhiLab/PyRawS/issues) to report a bug or to request a new feature.

If you want to contribute, please proceed as follow:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Create NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License
Distributed under the Apache License.

## Contacts
Created by the European Space Agency $\Phi$-[lab](https://phi.esa.int/).

* Gabriele Meoni - G.Meoni@tudelft.nl (previously, wit ESA $\Phi$-lab)
* Roberto Del Prete - roberto.delprete at ext.esa.int and unina.it
* Nicolas Longepe - nicolas.longepe at esa.int
* Federico Serva - federico.serva at ext.esa.int

## References
  ### [1]: [Massimetti, Francesco, et al. ""Volcanic hot-spot detection using SENTINEL-2: a comparison with MODIS–MIROVA thermal data series."" Remote Sensing 12.5 (2020): 820."](https://www.mdpi.com/2072-4292/12/5/820)