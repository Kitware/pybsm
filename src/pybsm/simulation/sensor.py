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


class Sensor:
    """Example details of the camera system.

    This is not intended to be a complete list but is more than adequate for the NIIRS demo (see
    pybsm.metrics.functional.niirs).

    :param name:
        name of the sensor
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)
    :param p_x:
        detector center-to-center spacings (pitch) in the x and y directions
        (meters); if p_y is not provided, it is assumed equal to p_x
    :param opt_trans_wavelengths:
        specifies the spectral bandpass of the camera (m); at minimum, specify
        a start and end wavelength
    :param optics_transmission:
        full system in-band optical transmission (unitless); do not include loss
        due to any telescope obscuration in this optical transmission array
    :param eta:
        relative linear obscuration (unitless); obscuration of the aperture
        commonly occurs within telescopes due to secondary mirror or spider
        supports
    :param p_y:
        detector center-to-center spacings (pitch) in the x and y directions
        (meters); if p_y is not provided, it is assumed equal to p_x
    :param w_x:
        detector width in the x and y directions (m); if set equal to p_x and
        p_y, this corresponds to an assumed full pixel fill factor. In general,
        w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
        (typically transistors) around each pixel.
    :param w_y:
        detector width in the x and y directions (m); if set equal to p_x and
        p_y, this corresponds to an assumed full pixel fill factor. In general,
        w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
        (typically transistors) around each pixel.
    :param int_time:
        maximum integration time (s)
    :param qe:
        quantum efficiency as a function of wavelength (e-/photon)
    :param qe_wavelengths:
        wavelengths corresponding to the array qe (m)
    :param other_irradiance:
        spectral irradiance from other sources (W/m^2 m); this is particularly
        useful for self emission in infrared cameras and may also represent
        stray light.
    :param dark_current:
        detector dark current (e-/s); dark current is the relatively small
        electric current that flows through photosensitive devices even when no
        photons enter the device
    :param max_n:
        detector electron well capacity (e-); the default 100 million
        initializes to a large number so that, in the absence of better
        information, it doesn't affect outcomes
    :param bit_depth:
        resolution of the detector ADC in bits (unitless); default of 100 is a
        sufficiently large number so that in the absence of better information,
        it doesn't affect outcomes
    :param n_tdi:
        number of TDI stages (unitless)
    :param cold_shield_temperature:
        temperature of the cold shield (K); it is a common approximation to
        assume that the coldshield is at the same temperature as the detector
        array
    :param optics_temperature:
        temperature of the optics (K)
    :param optics_emissivity:
        emissivity of the optics (unitless) except for the cold filter;
        a common approximation is 1-optics transmissivity
    :param cold_filter_transmission:
        transmission through the cold filter (unitless)
    :param cold_filter_temperature:
        temperature of the cold filter; it is a common approximation to assume
        that the filter is at the same temperature as the detector array
    :param cold_filter_emissivity:
        emissivity through the cold filter (unitless); a common approximation
        is 1-cold filter transmission
    :param s_x:
        root-mean-squared jitter amplitudes in the x direction (rad)
    :param s_y:
        root-mean-squared jitter amplitudes in the y direction (rad)
    :param da_x:
        line-of-sight angular drift rate during one integration time in the x
        direction (rad/s)
    :param da_y:
        line-of-sight angular drift rate during one integration time in the y
        direction (rad/s)
    :param pv:
        wavefront error phase variance (rad^2) -- tip: write as (2*pi*waves of
        error)^2
    :param pv_wavelength:
        wavelength at which pv is obtained (m)

    """

    # Correlation lengths of the phase autocorrelation function.  Apparently,
    # it is common to set the L_x to the aperture diameter.  (m)
    L_x: float

    # Correlation lengths of the phase autocorrelation function.  Apparently,
    # it is common to set the L_y to the aperture diameter.  (m)
    L_y: float

    # A catch all for noise terms that are not explicitly included elsewhere
    # (read noise, photon noise, dark current, quantization noise are
    # all already included)
    other_noise: np.ndarray

    # 2-D filter kernel (for sharpening or whatever).  Note that
    # the kernel is assumed to sum to one.
    filter_kernel: np.ndarray

    # The number of frames to be added together for improved SNR.
    frame_stacks: int

    def __init__(
        self,
        name: str,
        D: float,  # noqa: N803
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
        eta: float = 0.0,
        p_y: Optional[float] = None,
        w_x: Optional[float] = None,
        w_y: Optional[float] = None,
        int_time: float = 1,
        dark_current: float = 0,
        other_irradiance: float = 0.0,
        read_noise: float = 0,
        max_n: int = int(100.0e6),
        max_well_fill: float = 1.0,
        bit_depth: float = 100.0,
        n_tdi: float = 1.0,
        cold_shield_temperature: float = 70.0,
        optics_temperature: float = 270.0,
        optics_emissivity: float = 0.0,
        cold_filter_transmission: float = 1.0,
        cold_filter_temperature: float = 70.0,
        cold_filter_emissivity: float = 0.0,
        s_x: float = 0.0,
        s_y: float = 0.0,
        da_x: float = 0.0,
        da_y: float = 0.0,
        pv: float = 0.0,
        pv_wavelength: float = 0.633e-6,
    ) -> None:
        """Returns a sensor object whose name is *name* and...."""
        self.name = name
        self.D = D
        self.f = f
        self.p_x = p_x
        self.opt_trans_wavelengths = opt_trans_wavelengths
        self.optics_transmission = np.ones(opt_trans_wavelengths.shape[0])
        self.eta = eta

        if p_y is None:
            # Assume square pixels.
            self.p_y = p_x
        else:
            p_y = self.p_y

        if w_x is None:
            # Assume is 100% fill factor and square detectors.
            self.w_x = p_x
        else:
            self.w_x = w_x

        if w_y is None:
            # Assume is same fill factor as along the x.
            self.w_y = p_x / self.w_x * self.p_y
        else:
            self.w_y = w_y

        self.int_time = int_time
        self.dark_current = dark_current
        self.other_irradiance = other_irradiance
        self.read_noise = read_noise
        self.max_n = max_n
        self.max_well_fill = max_well_fill
        self.bit_depth = bit_depth
        self.n_tdi = n_tdi

        # TODO this should be exposed so a custom one can be provided.
        self.qe_wavelengths = opt_trans_wavelengths  # tplaceholder
        self.qe = np.ones(opt_trans_wavelengths.shape[0])  # placeholder

        # TODO I don't think these automatically get used everywhere they
        # should, some functions override by assuming different temperatures.
        self.cold_shield_temperature = cold_shield_temperature
        self.optics_temperature = optics_temperature
        self.optics_emissivity = optics_emissivity
        self.cold_filter_transmission = cold_filter_transmission
        self.cold_filter_temperature = cold_filter_temperature
        self.cold_filter_emissivity = cold_filter_emissivity
        self.s_x = s_x
        self.s_y = s_y
        self.da_x = da_x
        self.da_y = da_y
        self.pv = pv
        self.pv_wavelength = pv_wavelength
        self.L_x = D
        self.L_y = D
        self.other_noise = np.array([0])

        # TODO, before we expose these, we should track down whether they are
        # actually used anywhere downstream.
        self.filter_kernel = np.array([1])
        self.frame_stacks = 1
