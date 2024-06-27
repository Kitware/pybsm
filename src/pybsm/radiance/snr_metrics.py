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
import numpy as np


class SNRMetrics:
    """A generic class to fill with any outputs of interest."""

    opt_trans: np.ndarray
    qe: np.ndarray
    other_irradiance: np.ndarray
    tgt_n_rate: float
    tgt_FPA_irradiance: np.ndarray  # noqa: N815
    weights: np.ndarray
    tgt_n: float
    bkg_n_rate: float
    bkg_FPA_irradiance: np.ndarray  # noqa: N815
    bkg_n: float
    scale_factor: float
    int_time: float
    well_fraction: float
    contrast_signal: float
    signal_noise: float
    dark_current_noise: float
    quantization_noise: float
    self_emission_noise: float
    total_noise: float
    snr: float

    path_noise: float
    scene_noise: float

    def __init__(self, name: str) -> None:
        """Returns a sensor object whose name is *name*."""
        self.name = name
