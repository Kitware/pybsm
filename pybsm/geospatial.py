# -*- coding: utf-8 -*-
"""The Python Based Sensor Model (pyBSM) is a collection of electro-optical
camera modeling functions developed by the Air Force Research Laboratory,
Sensors Directorate.

Author citation:
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling
imaging systems", Proc. SPIE 10204 (2017)

Distribution A.  Approved for public release.
Public release approval for version 0.0: 88ABW-2017-3101
Public release approval for version 0.1: 88ABW-2018-5226

Maintainer: Kitware, Inc. <nrtk@kitware.com>
"""
# standard library imports
import os
import inspect
import warnings

# 3rd party imports
import numpy as np
from typing import Tuple

# new in version 0.2.  We filter warnings associated with calculations in the
# function circularApertureOTF.  These invalid values are caught as NaNs and
# appropriately replaced.
warnings.filterwarnings("ignore", r"invalid value encountered in arccos")
warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")
warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
warnings.filterwarnings("ignore", r"divide by zero encountered in true_divide")

# find the current path (used to locate the atmosphere database)
# dirpath = os.path.dirname(os.path.abspath(__file__))
dirpath = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

# define some useful physical constants
rEarth = 6378.164e3  # radius of the earth (m)


def altitudeAlongSlantPath(
    hTarget: float,
    hSensor: float,
    slantRange: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the height above the curved earth at points along a path from the
    target (zPath=0) to the sensor (zPath.max()).  This is primarily useful for
    calculating the atmospheric coherence diameter, r0.

    :param hTarget:
        height of the target above sea level (m)
    :param hSensor:
        height of the sensor above sea level (m)
    :param slantRange:
        distance between the target and sensor (m)

    :return:
        zPath:
            array of samples along the path from the target (zPath = 0) to the
            sensor (m)
        hPath:
            height above the earth along a slantpath defined by zPath (m)

    :raises:
        ZeroDivisionError:
            if slantRange is 0

    """

    # this is simple law of cosines problem
    nadir = nadirAngle(hTarget, hSensor, slantRange)
    a = rEarth + hSensor
    b = np.linspace(
        0.0, slantRange, 10000
    )  # arbitrary choice of 100,000 data points
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(nadir))

    zPath = b
    hPath = (c - rEarth)[::-1]
    # It is correct to reverse the order of hPath.  The height of the target
    # above the earth (hTarget) should be the first element in the array.
    # Effectively, this reversal just changes the location of the origin of
    # zPath to the target location.  It was just more convenient to use the
    # origin at the sensor in the law of cosines calculation.

    return (zPath, hPath)


def groundSampleDistance(
    ifov: float,
    slantRange: float
) -> float:
    """IBSM Equation 3-62.  The ground sample distance, i.e. the footprint
    of a single detector in object space.

    :param ifov:
        instantaneous field-of-view of a detector (radians)
    :param slantRange:
        slant range to target (m)

    :return:
        gsd:
            ground sample distance (m)
    """
    gsd = slantRange * ifov
    return gsd


def nadirAngle(
    hTarget: float,
    hSensor: float,
    slantRange: float
) -> float:
    """
    Work through the law of cosines to calculate the sensor nadir angle above
    a circular earth. (i.e. angle between looking straight down (nadir = 0) and looking along the
    slant path).

    :param hTarget:
        height of the target above sea level (m)
    :param hSensor:
        height of the sensor above sea level (m)
    :param slantRange:
        distance between the target and sensor (m)

    :return:
        nadir:
            the sensor nadir angle (rad)

    :raises:
        ZeroDivisionError:
            if slantRange is 0
    """

    a = rEarth + hSensor
    b = slantRange
    c = rEarth + hTarget
    nadir = np.arccos(-1.0 * (c**2.0 - a**2.0 - b**2.0) / (2.0 * a * b))

    return nadir


def curvedEarthSlantRange(
    hTarget: float,
    hSensor: float,
    groundRange: float
) -> float:
    """Returns the slant range from target to sensor above a curved (circular)
    Earth.

    :param hTarget:
        height of the target above sea level (m)
    :param hSensor:
        height of the sensor above sea level (m)
    :param groundRange:
        distance between the target and sensor on the ground (m)

    :return:
        slantRange:
            distance between the target and sensor (m)
    """
    a = rEarth + hSensor
    c = rEarth + hTarget
    theta = groundRange / rEarth  # exact arc length angle (easy to derive)
    slantRange = np.sqrt(c**2.0 + a**2.0 - 2.0 * a * c * np.cos(theta))

    return slantRange
