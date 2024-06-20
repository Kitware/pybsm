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
# 3rd party imports
import numpy as np


class OTF:
    """Simple object to hold all the of the OTFs."""
    apOTF: np.ndarray
    turbOTF: np.ndarray
    r0band: np.ndarray
    detOTF: np.ndarray
    jitOTF: np.ndarray
    drftOTF: np.ndarray
    wavOTF: np.ndarray
    filterOTF: np.ndarray
    systemOTF: np.ndarray

    def __init__(self) -> None:
        pass
