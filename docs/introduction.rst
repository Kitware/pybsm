Introduction
============

Welcome to the documentation for the Python Based Sensor Model (pyBSM), a platform created for
developers seeking to simulate the image formation process through differing imaging systems. Its 
main purpose is to create realistic, augmented images from existing imagery. This 
package builds off of the work done by LeMaster and Eismann in creating the original pyBSM package 
for modeling image systems [1] as well as LeMaster et al. in their work in validating the pyBSM 
package [2].

This documentation is structured to provide you with straightforward and practical instructions and
examples, so that you can effectively leverage pyBSM to create realistic, augmented images.

Background
----------

Creating a well-made model from a dataset requires a myriad of images. For satellite images, datasets tend 
to be limited not only in the number of images but also the variety of images. One possible solution
is to apply augmentations. In most computer vision applications, applying augmentations such as
blurring or adding noise can be used to assess a model's robustness to different perturbations;
however, in satellite imagery, performing more complex augmentations requires a detailed sensor
modeling framework. pyBSM offers a possible solution
with realistic, augmented images.

At the core, pyBSM differs from existing image augmentation libraries (e.g. `imgaug <https://github.com/aleju/imgaug>`_)
by using physics-based, sensor-specific perturbations. Rather than applying a generic augmentation 
to an image, pyBSM modifies known parameters to create an augmented image with the new set of
parameters. Ideally, the augmented images would be equivalent to a new set of images taken in the 
same state as the modified parameters. pyBSM is not only able to increase the number of images in a 
dataset, but also create realistic images that enhance model performance. 

Use Cases
---------

At a high level, pyBSM can create synthetic images based on sensor parameters (e.g. focal length, 
aperture, pixel pitch, etc.) and scenario parameters (e.g. altitude, ground range, visibility, etc.). 
Starting with an aerial image with known sensor and scenario parameters, pyBSM can generate imagery taken
from a hypothetical sensor with a different altitude, focal length, or any combination of sensor and 
scenario parameters. As mentioned previously, the main use case is to improve the coverage of existing datasets to relevant sensor and scenario parameters,
which can be used to assess the robustness of computer vision models.


References
----------

1. LeMaster, Daniel A., and Michael T. Eismann. 2017. "pyBSM: A Python package for modeling imaging
systems." Proceedings of the SPIE 10204.

2. LeMaster, Daniel A., et al. Validating pyBSM: A Python Package for Modeling Imaging Systems. 05 2018, p. 19, https://doi.org10.1117/12.2305228.

