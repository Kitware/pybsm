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
from typing import Optional


class Sensor:
    """Example details of the camera system.  This is not intended to be a
    complete list but is more than adequate for the NIIRS demo (see
    pybsm.niirs).

    :param name:
        Name of the sensor
    :param D:
        Effective aperture diameter (m)
    :param f:
        Focal length (m)
    :param px:
        Detector center-to-center spacings (pitch) in the x and y directions
        (meters). IF py is not provided, it is assumed equal to px.
    :param optTransWavelengths:
        Specifies the spectral bandpass of the camera (m).  At minimum, and
        start and end wavelength should be specified.
    :param opticsTransmission:
        Full system in-band optical transmission (unitless).  Loss due to any
        telescope obscuration should *not* be included in with this optical
        transmission array.
    :param eta:
        Relative linear obscuration (unitless). Obscuration of the aperture
        commonly occurs within telescopes due to secondary mirror or spider
        supports.
    :param py:
        Detector center-to-center spacings (pitch) in the x and y directions
        (meters). IF py is not provided, it is assumed equal to px.
    :param wx:
        Detector width in the x and y directions (m). If set equal to px and
        py, this corresponds to an assumed full pixel fill factor. In general,
        wx and wy are less than px and py due to non-photo-sensitive area
        (typically transistors) around each pixel.
    :param wy:
        Detector width in the x and y directions (m). If set equal to px and
        py, this corresponds to an assumed full pixel fill factor. In general,
        wx and wy are less than px and py due to non-photo-sensitive area
        (typically transistors) around each pixel.
    :param qe:
        Quantum efficiency as a function of wavelength (e-/photon).
    :param qewavelengths:
        Wavelengths corresponding to the array qe (m).
    :param otherIrradiance:
        Spectral irradiance from other sources (W/m^2 m). This is particularly
        useful for self emission in infrared cameras.  It may also represent
        stray light.
    :param darkCurrent:
        Detector dark current (e-/s). Dark current is the relatively small
        electric current that flows through photosensitive devices even when no
        photons enter the device.
    :param maxN:
        Detector electron well capacity (e-). The default 100 million,
        initializes to a large number so that, in the absence of better
        information, it doesn't affect outcomes.
    :param bitdepth:
        Resolution of the detector ADC in bits (unitless). Default of 100 is
        sufficiently large number so that in the absense of better information,
        it doesn't affect outcomes.
    :param ntdi:
        Number of TDI stages (unitless).
    :param coldshieldTemperature:
        Temperature of the cold shield (K).  It is a common approximation to
        assume that the coldshield is at the same temperature as the detector
        array.
    :param opticsTemperature:
        Temperature of the optics (K)
    :param opticsEmissivity:
        Emissivity of the optics (unitless) except for the cold filter.
        A common approximation is 1-optics transmissivity.
    :param coldfilterTransmission:
        Transmission through the cold filter (unitless)
    :param coldfilterTemperature:
        Temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.
    :param coldfilterEmissivity:
        Emissivity through the cold filter (unitless).  A common approximation
        is 1-cold filter transmission
    :param sx:
        Root-mean-squared jitter amplitudes in the x direction. (rad)
    :param sy:
        Root-mean-squared jitter amplitudes in the y direction. (rad)
    :param dax:
        Line-of-sight angular drift rate during one integration time in the x
        direction. (rad/s)
    :param day:
        Line-of-sight angular drift rate during one integration time in the y
        direction. (rad/s)
    :param pv:
        Wavefront error phase variance (rad^2) - tip: write as (2*pi*waves of
        error)^2
    :param pvwavelength:
        Wavelength at which pv is obtained (m)

    """

    # Correlation lengths of the phase autocorrelation function.  Apparently,
    # it is common to set the Lx to the aperture diameter.  (m)
    Lx: float

    # Correlation lengths of the phase autocorrelation function.  Apparently,
    # it is common to set the Ly to the aperture diameter.  (m)
    Ly: float

    # A catch all for noise terms that are not explicitly included elsewhere
    # (read noise, photon noise, dark current, quantization noise are
    # all already included)
    otherNoise: np.ndarray

    # 2-D filter kernel (for sharpening or whatever).  Note that
    # the kernel is assumed to sum to one.
    filterKernel: np.ndarray

    # The number of frames to be added together for improved SNR.
    framestacks: int

    def __init__(
        self,
        name: str,
        D: float,
        f: float,
        px: float,
        optTransWavelengths: np.ndarray,
        eta: float = 0.0,
        py: Optional[float] = None,
        wx: Optional[float] = None,
        wy: Optional[float] = None,
        intTime: float = 1,
        darkCurrent: float = 0,
        otherIrradiance: float = 0.0,
        readNoise: float = 0,
        maxN: int = int(100.0e6),
        maxWellFill: float = 1.0,
        bitdepth: float = 100.0,
        ntdi: float = 1.0,
        coldshieldTemperature: float = 70.0,
        opticsTemperature: float = 270.0,
        opticsEmissivity: float = 0.0,
        coldfilterTransmission: float = 1.0,
        coldfilterTemperature: float = 70.0,
        coldfilterEmissivity: float = 0.0,
        sx: float = 0.0,
        sy: float = 0.0,
        dax: float = 0.0,
        day: float = 0.0,
        pv: float = 0.0,
        pvwavelength: float = 0.633e-6
    ) -> None:
        """Returns a sensor object whose name is *name* and...."""
        self.name = name
        self.D = D
        self.f = f
        self.px = px
        self.optTransWavelengths = optTransWavelengths
        self.opticsTransmission = np.ones(optTransWavelengths.shape[0])
        self.eta = eta

        if py is None:
            # Assume square pixels.
            self.py = px
        else:
            py = self.py

        if wx is None:
            # Assume is 100% fill factor and square detectors.
            self.wx = px
        else:
            self.wx = wx

        if wy is None:
            # Assume is same fill factor as along the x.
            self.wy = px / self.wx * self.py
        else:
            self.wy = wy

        self.intTime = intTime
        self.darkCurrent = darkCurrent
        self.otherIrradiance = otherIrradiance
        self.readNoise = readNoise
        self.maxN = maxN
        self.maxWellFill = maxWellFill
        self.bitdepth = bitdepth
        self.ntdi = ntdi

        # TODO this should be exposed so a custom one can be provided.
        self.qewavelengths = optTransWavelengths  # tplaceholder
        self.qe = np.ones(optTransWavelengths.shape[0])  # placeholder

        # TODO I don't think these automatically get used everywhere they
        # should, some functions overridde by assuming different temperatures.
        self.coldshieldTemperature = coldshieldTemperature
        self.opticsTemperature = opticsTemperature
        self.opticsEmissivity = opticsEmissivity
        self.coldfilterTransmission = coldfilterTransmission
        self.coldfilterTemperature = coldfilterTemperature
        self.coldfilterEmissivity = coldfilterEmissivity
        self.sx = sx
        self.sy = sy
        self.dax = dax
        self.day = day
        self.pv = pv
        self.pvwavelength = pvwavelength
        self.Lx = D
        self.Ly = D
        self.otherNoise = np.array([0])

        # TODO, before we expose these, we should track down whether they are
        # actually used anywhere downstream.
        self.filterKernel = np.array([1])
        self.framestacks = 1
