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
from typing import Tuple

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

# define some useful physical constants
r_earth = 6378.164e3  # radius of the earth (m)


def altitude_along_slant_path(
    h_target: float, h_sensor: float, slant_range: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the height above the curved earth at points along a path from the target to the sensor.

    Calculate the height above the curved earth at points along a path from the
    target (z_path=0) to the sensor (z_path.max()).  This is primarily useful for
    calculating the atmospheric coherence diameter, r0.

    :param h_target:
        height of the target above sea level (m)
    :param h_sensor:
        height of the sensor above sea level (m)
    :param slant_range:
        distance between the target and sensor (m)

    :return:
        z_path:
            array of samples along the path from the target (z_path = 0) to the
            sensor (m)
        h_path:
            height above the earth along a slantpath defined by z_path (m)

    :raises:
        ZeroDivisionError:
            if slant_range is 0

    """
    # this is simple law of cosines problem
    nadir = nadir_angle(h_target, h_sensor, slant_range)
    a = r_earth + h_sensor
    b = np.linspace(0.0, slant_range, 10000)  # arbitrary choice of 100,000 data points
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(nadir))

    z_path = b
    h_path = (c - r_earth)[::-1]
    # It is correct to reverse the order of h_path.  The height of the target
    # above the earth (h_target) should be the first element in the array.
    # Effectively, this reversal just changes the location of the origin of
    # z_path to the target location.  It was just more convenient to use the
    # origin at the sensor in the law of cosines calculation.

    return (z_path, h_path)


def ground_sample_distance(ifov: float, slant_range: float) -> float:
    """IBSM Equation 3-62, the ground sample distance.

    The ground sample distance, i.e. the footprint of a single detector in object space.

    :param ifov:
        instantaneous field-of-view of a detector (radians)
    :param slant_range:
        slant range to target (m)

    :return:
        gsd:
            ground sample distance (m)
    """
    gsd = slant_range * ifov
    return gsd


def nadir_angle(h_target: float, h_sensor: float, slant_range: float) -> float:
    """Work through the law of cosines to calculate the sensor nadir angle above a circular earth.

    Work through the law of cosines to calculate the sensor nadir angle above a
    circular earth (i.e. angle between looking straight down (nadir = 0) and looking
    along the slant path).

    :param h_target:
        height of the target above sea level (m)
    :param h_sensor:
        height of the sensor above sea level (m)
    :param slant_range:
        distance between the target and sensor (m)

    :return:
        nadir:
            the sensor nadir angle (rad)

    :raises:
        ZeroDivisionError:
            if slant_range is 0
    """
    a = r_earth + h_sensor
    b = slant_range
    c = r_earth + h_target
    nadir = np.arccos(-1.0 * (c**2.0 - a**2.0 - b**2.0) / (2.0 * a * b))

    return nadir


def curved_earth_slant_range(
    h_target: float, h_sensor: float, ground_range: float
) -> float:
    """Returns the slant range from target to sensor above a curved (circular) Earth.

    :param h_target:
        height of the target above sea level (m)
    :param h_sensor:
        height of the sensor above sea level (m)
    :param ground_range:
        distance between the target and sensor on the ground (m)

    :return:
        slant_range:
            distance between the target and sensor (m)
    """
    a = r_earth + h_sensor
    c = r_earth + h_target
    theta = ground_range / r_earth  # exact arc length angle (easy to derive)
    slant_range = np.sqrt(c**2.0 + a**2.0 - 2.0 * a * c * np.cos(theta))

    return slant_range
