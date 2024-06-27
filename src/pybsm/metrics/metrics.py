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
# 3rd party imports
from typing import Optional

import numpy as np

# local imports
import pybsm.otf as otf
from pybsm.radiance import SNRMetrics
from pybsm.simulation import Scenario, Sensor


class Metrics:
    """A generic class to fill with any outputs of interest."""

    sensor: Sensor
    scenario: Scenario
    slant_range: float
    atm: np.ndarray
    tgt_radiance: np.ndarray
    bkg_radiance: np.ndarray
    radiance_wavelengths: np.ndarray
    snr: SNRMetrics
    mtf_wavelengths: np.ndarray
    mtf_weights: np.ndarray
    cutoff_frequency: float
    uu: np.ndarray
    vv: np.ndarray
    df: float
    otf: otf.OTF
    ifov_x: float
    ifov_y: float
    gsd_x: float
    gsd_y: float
    gsd_gm: float
    rer_gm: float
    eho_gm: float
    ng: float
    niirs: float
    elev_angle: float
    niirs_4: float
    gsd_gp: float
    rer: float
    rer_0: float
    rer_90: float
    gsd_w: float

    def __init__(self, name: Optional[str]) -> None:
        """Returns a sensor object whose name is *name*."""
        self.name = name
