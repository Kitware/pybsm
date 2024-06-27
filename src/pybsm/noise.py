# -*- coding: utf-8 -*-
"""The Python Based Sensor Model (pyBSM) is a collection of electro-optical camera modeling functions.

Developed by the Air Force Research Laboratory, Sensors Directorate.

Author citation:
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling
imaging systems", Proc. SPIE 10204 (2017)

Distribution A.  Approved for public release.
Public release approval for version 0.0: 88ABW-2017-3101
Public release approval for version 0.1: 88ABW-2018-5226

Maintainer: Kitware, Inc. <nrtk@kitware.com>
"""
# standard library imports
import inspect
import os
import warnings

# 3rd party imports
import numpy as np

# new in version 0.2.  We filter warnings associated with calculations in the
# function circularApertureOTF.  These invalid values are caught as NaNs and
# appropriately replaced.
warnings.filterwarnings("ignore", r"invalid value encountered in arccos")
warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")
warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
warnings.filterwarnings("ignore", r"divide by zero encountered in true_divide")

# find the current path (used to locate the atmosphere database)
# dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def noise_gain(kernel: np.ndarray) -> float:
    """Noise Gain is the GIQE term representing increase in noise due to image sharpening.

    The definition is not included in the IBSM manual.  This version comes from
    Leachtenauer et al., "General Image-Quality Equation: GIQE" APPLIED OPTICS
    Vol. 36, No. 32 10 November 1997.

    :param kernel:
         the 2D image sharpening kernel; note that
         the kernel is assumed to sum to one

    :return:
        ng:
            noise gain (unitless)
    """
    ng = np.sqrt(np.sum(np.sum(kernel**2)))
    return ng


def quantization_noise(pe_range: float, bit_depth: float) -> float:
    """Effective noise contribution from the number of photoelectrons quantized.

    Effective noise contribution from the number of photoelectrons quantized by a single count of the analog
    to digital converter.Quantization noise is buried in the definition of signal-to-noise in IBSM equation 3-47.

    :param pe_range:
        the difference between the maximum and minimum number of photoelectrons
        that may be sampled by the readout electronics (e-)
    :param bit_depth:
        number of bits in the analog to digital converter (unitless)

    :return:
        sigma_q :
            quantization noise given as a photoelectron standard deviation (e-)

    :WARNING:
        output can be nan if pe_range is 0
        output can be inf if bit_depth is 0
    """
    sigma_q = pe_range / (np.sqrt(12) * (2.0**bit_depth - 1.0))
    return sigma_q
