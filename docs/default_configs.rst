.. _flir blackfly s: https://www.eureca.de/files/pdf/optoelectronics/flir/BFS-PGE-23S3-Datasheet.pdf

.. _pybsm/docs/explanation.rst: explanation.html

.. _pybsm/docs/miscellaneous/tank_image.jpg: miscellaneous/tank_image.html

.. _pybsm/docs/miscellaneous/uav_default_config.json: miscellaneous/uav_default_config.json

.. _pybsm/docs/useful_calculations.rst: useful_calculations.html

####################################################
 Default PyBSM Sensor and Simulation Configurations
####################################################

Fully and properly utilizing pyBSM requires careful selection of multiple
sensor, scenario, and image variables. Altering these variables can have
significant effects on the pyBSM simulations and the resulting imagery. This
document aims to explore several example configurations that will help a user
get started in their relevant domain and also serve as a foundation for
application-specific changes to be made.

For more information about these variables, please visit the documentation in
`pybsm/docs/explanation.rst`_. For some useful calculations for these variables,
please visit the documentation in `pybsm/docs/useful_calculations.rst`_.

*****************************
 Common Parameters to Adjust
*****************************

If you have a default config (or want to use the UAV config explained below),
there are several parameters that are common to adjust based on the specific
scenario or sensor being used.

Ground-Sampling Distance ``gsd``
================================

GSD defines how much of the physical scene each pixel contains. Please refer to
`pybsm/docs/useful_calculations.rst`_ for two different ways to estimate the GSD
depending on the available information.

Altitude ``h``
==============

This should be calculated/estimated very similarily to how you calculate the GSD
in the above calculation. Again, refer to `pybsm/docs/useful_calculations.rst`_
for how altitude affects GSD.

Lens Focal Length ``f``
=======================

The focal length mainly determines the field-of-view (FOV) of the sensor, where
a shorter ``f`` results in a wider FOV and vice-versa. Most machine vision focal
lengths are between 12mm-100mm, but anything longer than 30mm is going to be
fairly uncommon since that requires a very large optics.
`pybsm/docs/useful_calculations.rst`_ explains how FOV is related to the focal
length.

Lens Diameter ``D``
===================

The lens diameter is important for how much light enters the lens, the
depth-of-field, and some of the diffraction-limited PSFs. This is most commonly
calculated via the **f-number (F#)**, which describes the diameter in terms of
the focal length ``f``. For example, F# of ``f``/4 for a 35mm focal length lens
means the diameter ``D`` is 35/4 = 8.75mm, while a F# of ``f``/1.4 means ``D``
is 25mm. It is pretty rare for a f number to go below ``f``/1.4 and above
``f``/16, where a higher F# means a smaller diameter lens.

Sensor's Bit Depth ``bit_depth``
================================

This usually can be deduced by looking at the maximum pixel value in your
images, as the number of bits must be at least as large as that number. Note
that bit depth represents the number of bits used, **not** the total number of
values. The total range is given by :math:`2^{\text{bit_depth}}`. For example,
if the maximum pixel value in the image is 985, I would estimate a bit depth of
"10" since :math:`2^{10}=1024 > 985`. ``bit_depth`` should be an integer and is
typically either 8, 10, 12, or 16.

Max Number of Well Electrons ``max_n``
======================================

This describes how many electrons a pixel can accumulate. More electrons means
more dynamic range, while less means faster sensor saturation. Realistically,
the typical visible sensor has a ``max_n`` around 4-10ke, short wave infrared
(SWIR) is 20-50ke, medium wave MWIR is 1-4Me, long wave LWIR is 5-200Me.

However, sometimes we are focused on simulating the effects of high exposure
times or pixel drifts ``da_x``. In that case or similar cases, you can increase
``max_n`` to an abitrarily large amount (such as > 100e6). This will harm the
realism of the SNR but allow for more of these simulated effects.

***************************
 Example UAV Configuration
***************************

The following is a configuration specialized for a UAV imaging a scene at nadir
in which we know the sensor. We assume that we are using the `FLIR Blackfly S`_
CMOS sensor and that the UAV is at an altitude of 100m above the imaging plane
traveling at 15 m/s. We also assume that the image's FOV on the ground is
approximately a 28m x 17m box. The example image is of a tank. A json containing
these example configs is in `pybsm/docs/miscellaneous/uav_default_config.json`_
and the example reference tank image is in
`pybsm/docs/miscellaneous/tank_image.jpg`_.


**Sensor Variables** Based on our given sensor (`FLIR Blackfly S`_), we can
notice that the underlying sensor is a Sony IMX392. This sensor has some
additional information available `here
<https://softwareservices.flir.com/BFS-U3-23S3/latest/EMVA/EMVA.html>`_ about
its spectral response and noise. Based on the documents and our UAV scenario, we
can extact the parameters:

-  ``p_x`` = 3.45e-6

-  ``p_y`` = 3.45e-6

-  ``w_x`` = 3.45e-6 (not given, but the fill factor should be nearly 1 because
   of BSI pixel architecture)

-  ``w_y`` = 3.45e-6

-  ``f`` = 24e-3 (corresponds to the 28m x 17m imaging box which is a 15.6 x 9.7
   degrees FOV. For more information how to calculate this, look at the
   documentation in `pybsm/docs/useful_calculations.rst`_.)

-  ``D`` = 8.75e-3 (corresponds to f-number of f/4. Typically, we want )

-  ``int_time`` = 17e-3 (corresponds to 60 FPS)

-  ``bit_depth`` = 12

-  ``dark_current`` = 0.0 (none given, so default to 0)

-  ``max_n`` = 11105

-  ``s_x`` = 8e-3 (corresponds to a RMS jitter of 0.5 deg)

-  ``s_y`` = 8e-3

-  ``da_x`` = 0.0 (corresponds to travel entirely in y direction)

-  ``da_y`` = 2.5e-3 (corresponds to travel entirely in y direction. For more
   information on how to calculate angular drift, look at the documentation in
   `pybsm/docs/useful_calculations.rst`_ )

-  ``qewavelengths`` =[470e-9, 525e-9, 630e-9, 800e-9]

-  ``qe`` =[0.50, 0.60, 0.48, 0.25]

-  ``opt_trans_wavelengths`` [470e-9, 800e-9]

**Scenario Variables** These variables are highly reliant on our scenario, that
is, a UAV 100m above ground level traveling at 15 m/s and the corresponding wind
environment.

-  ``ihaze``: 1
-  ``altitude``: 100
-  ``ground_range``: 0 (corresponds to imaging directly below UAV)
-  ``aircraft_speed``: 15
-  ``target_reflectance``: 0.15
-  ``background_reflectance``: 0.07
-  ``target_temperature``: 305
-  ``background_temperature``: 300
-  ``ha_wind_speed``: 8 (corresponds to a medium wind strength condition)
-  ``cn2_at_1m``: 1.7e-14
-  ``interp``: False

**Reference Image Variables** While these variables rely on the actual reference
image, we will show the corresponding values that would approximately be
captured by the sensor and scenario we have in place.

-  ``img``: actual pixel values (in this case,
   `pybsm/docs/miscellaneous/tank_image.jpg`_)

-  ``gsd``: 14.6e-3

-  ``pix_values``: [0, 128, 256, 384, 512, 640, 768, 896 1024, 1152, 1280, 1408,
   1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
   3200, 3328, 3456, 3584, 3712, 3840, 3968, 4095]

-  ``refl_values``: [0.0e+00, 4.9e-04, 2.2e-03, 5.5e-03, 1.0e-02, 1.7e-02,
   2.5e-02, 3.5e-02, 4.7e-02, 6.1e-02, 7.7e-02, 9.5e-02, 1.2e-01, 1.4e-01,
   1.6e-01, 1.9e-01, 2.2e-01, 2.5e-01, 2.8e-01, 3.2e-01, 3.6e-01, 4.0e-01,
   4.4e-01, 4.8e-01, 5.3e-01, 5.8e-01, 6.3e-01, 6.9e-01, 7.5e-01, 8.1e-01,
   8.7e-01, 9.3e-01, 1.0] (Assuming max pixel value corresponds to reflectance
   of 1. For more information on how to calculate this, look at the
   documentation in `pybsm/docs/useful_calculations.rst`_)
