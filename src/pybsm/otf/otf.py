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


class OTF:
    """Simple object to hold all the of the OTFs."""

    ap_OTF: np.ndarray  # noqa: N815
    turb_OTF: np.ndarray  # noqa: N815
    r0_band: np.ndarray
    det_OTF: np.ndarray  # noqa: N815
    jit_OTF: np.ndarray  # noqa: N815
    drft_OTF: np.ndarray  # noqa: N815
    wav_OTF: np.ndarray  # noqa: N815
    filter_OTF: np.ndarray  # noqa: N815
    system_OTF: np.ndarray  # noqa: N815

    def __init__(self) -> None:
        pass
