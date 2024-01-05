# -*- coding: utf-8 -*-
"""The python Based Sensor Model (pyBSM) is a collection of electro-optical
camera modeling functions developed by the Air Force Research Laboratory,
Sensors Directorate.

Please use the following citation:
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling
imaging systems", Proc. SPIE 10204 (2017)

Distribution A.  Approved for public release.
Public release approval for version 0.0: 88ABW-2017-3101
Public release approval for version 0.1: 88ABW-2018-5226


contact: daniel.lemaster@us.af.mil

version 0.2: CURRENTLY IN BETA!!
"""
# 3rd party imports
import numpy as np


class SNRMetrics:
    """A generic class to fill with any outputs of interest."""
    optTrans: np.ndarray
    qe: np.ndarray
    otherIrradiance: np.ndarray
    tgtNrate: float
    tgtFPAirradiance: np.ndarray
    weights: np.ndarray
    tgtN: float
    bkgNrate: float
    bkgFPAirradiance: np.ndarray
    bkgN: float
    scalefactor: float
    intTime: float
    wellfraction: float
    contrastSignal: float
    signalNoise: float
    darkcurrentNoise: float
    quantizationNoise: float
    selfEmissionNoise: float
    totalNoise: float
    snr: float

    pathNoise: float
    sceneNoise: float

    def __init__(
        self,
        name: str
    ) -> None:
        """Returns a sensor object whose name is *name* """
        self.name = name
