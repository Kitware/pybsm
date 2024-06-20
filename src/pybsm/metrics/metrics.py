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
from typing import Optional

# local imports
import pybsm.otf as otf
from pybsm.radiance import SNRMetrics
from pybsm.simulation import Scenario, Sensor


class Metrics:
    """A generic class to fill with any outputs of interest."""

    sensor: Sensor
    scenario: Scenario
    slantRange: float
    atm: np.ndarray
    tgtRadiance: np.ndarray
    bkgRadiance: np.ndarray
    radianceWavelengths: np.ndarray
    snr: SNRMetrics
    mtfwavelengths: np.ndarray
    mtfweights: np.ndarray
    cutoffFrequency: float
    uu: np.ndarray
    vv: np.ndarray
    df: float
    otf: otf.OTF
    ifovx: float
    ifovy: float
    gsdx: float
    gsdy: float
    gsdgm: float
    rergm: float
    ehogm: float
    ng: float
    niirs: float
    elevAngle: float
    niirs4: float
    gsdgp: float
    rer: float
    rer0: float
    rer90: float
    gsdw: float

    def __init__(
        self,
        name: Optional[str]
    ) -> None:
        """Returns a sensor object whose name is *name* """
        self.name = name
