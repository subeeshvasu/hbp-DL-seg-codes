# Deep Learning Frameworks for HBP

This repo contains segmentation routines associated with supervised and semi-supervised learning schemes. All methods are developed using variants of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). Developed at [CVLab@EPFL](https://cvlab.epfl.ch/) as part of our work for Task 5.6.3 of the Human Brain Project (SGA2). The ultimate goal is to develop tools for automated processing of medical images and integrate them into [ilastik](ilastik.org).

List of dependencies can be found in reqs.txt.

## To run the test code

* Downlaod the pretrained weights and save them in respective folders (``weights/<network name>/``). Links to the pretrained weights are provided in ``weights/<network name>/``.

* To obtain the results of 2sUnet and Unet, run the scripts test_unet_DA.py and test_2sunet_DA.py respectively (example: python test_unet_DA.py). The code will perform inference on the sample images from ``sample_test_images/`` (data from Ludovico Silvestri's European Laboratory for Non-linear Spectroscopy (LENS)) and the results will get saved in ``testouts/``. 
