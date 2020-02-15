# Deep Learning Frameworks for HBP

This repo contains segmentation routines associated with supervised and semi-supervised learning schemes. All methods are developed using variants of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). Developed at [CVLab@EPFL](https://cvlab.epfl.ch/) as part of our work for Task 5.6.3 of the Human Brain Project (SGA2). The ultimate goal is to develop tools for automated processing of medical images and integrate them into [ilastik](ilastik.org). 

The implementation allows for 2D or 3D filters. Settings are specified via configuration files: see `config/spec.cfg` for options and other `.cfg` files for examples. Datasets are not made public as they are undergoing curation, but they should be in the future (please do inquire). Using your own data should be easy: follow the examples on `dataset.py` to format it as lists of slices (2D) or stacks (3D). This code is provided as-is, without further support.

Requires pytorch 0.3. For an overcomplete list of dependencies, please check `reqs.txt`.
