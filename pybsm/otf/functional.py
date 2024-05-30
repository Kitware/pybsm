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

This module contains a library of functions implementing the optical transfer
function (OTF) for various modes of sharpness degradation in an optical system.
The full system-level OTF is created by convolving all of the component-level
OTF results. Having this system-level OTF for the imaging system we are
modeling, we can take a notional ideal image and convolve it with the OTF to
produce a high-accuracy estimate for what that image would look like if imaged
by this our modeled imaging system.

Such a treatment is based on the premise that we can model an optical system as
a linear spatial-invariant system completely defined by a system-level OTF.
Here, spatial-invariance is referring to the fact that we assume the OTF is
constant across the field of view. This framework is motivated by the field of
Fourier optics.

Due to many different factors, all real-world optical systems produce an
imperfect image of the world they are observing. One form of degradation is
geometric distortion (barrel, pincushion), where features in the image are
stretched to different locations, and another aspect is the addition of random
or fixed-pattern noise. However, these modes of degradation are not considered
by this module or by the treatment with the system-level OTF. The OTF treatment
deals with, loosely speaking, all modes of degradation that less to reduction
in spatial resolution or sharpness.

The simplest example that we can consider to understand what the OTF
represents is the scenario of imaging a distant star with our imaging system.
For all intents and purposes, a star is an infinitesimal point source of light
(i.e., perfect plane wave incident on the optical system's aperture plane), and
the perfect image of the star would also be a point of light. However, the
actual image of that star created by the imaging system will always be some
blurred-out extended shape, and that shape is by definition the OTF. For
example, the highest-quality imaging systems are often "diffraction-limited",
meaning all other aspects of the imaging system were sufficiently optimized
such that its resolution is defined by the fundamental limit imposed by
diffraction, the OTF is an Airy disk shape, and the angular resolution is
approximately 1.22*lambda/d, where lambda is the wavelength of the light and d
is the aperture diameter.

All of the functions in this module that end with OTF are of the form

        H = <degradation-type>OTF(u, v, extra_parameters)

where u and v are the horizontal and vertical angular spatial frequency
coordinates (rad^-1) and 'extra_parameters' captures all of relevant parameters
of the imaging system dictating the particular mode of OTF. The return, H, is
the OTF response (unitless) for that those spatial frequencies.
"""
# standard library imports
import os
import inspect
import warnings

# 3rd party imports
import numpy as np
from scipy import interpolate
from scipy.special import jn
from typing import Callable, Tuple
import cv2

# local imports
from pybsm.geospatial import altitudeAlongSlantPath
from pybsm.simulation.sensor import Sensor
from pybsm.simulation.scenario import Scenario
from .otf import OTF


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
hc = 6.62607004e-34  # Plank's constant  (m^2 kg / s)
cc = 299792458.0  # speed of light (m/s)
kc = 1.38064852e-23  # Boltzmann constant (m^2 kg / s^2 K)
qc = 1.60217662e-19  # charge of an electron (coulombs)
rEarth = 6378.164e3  # radius of the earth (m)


# ------------------------------- OTF Models ---------------------------------

def circularApertureOTF(
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    D: float,
    eta: float
) -> np.ndarray:
    """IBSM Equation 3-20.  Obscured circular aperture diffraction OTF.  If eta
    is set to 0, the function will return the unobscured aperture result.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param lambda0:
        wavelength (m)
    :param D:
        effective aperture diameter (m)
    :param eta:
        relative linear obscuration (unitless)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    :raises:
        ZeroDivisionError:
            if lambda0 is 0

    :WARNING:
        Output can be nan if eta is 1.

    :NOTE:
        You will see several runtime warnings when this code is first accessed.
        The issue (calculating arccos and sqrt outside of their domains) is
        captured and corrected np.nan_to_num
    """

    rho = np.sqrt(u**2.0 + v**2.0)  # radial spatial frequency
    r0 = D / lambda0  # diffraction limited cutoff spatial frequency (cy/rad)

    # this A term by itself is the unobscured circular aperture OTF
    A = (2.0 / np.pi) * (
        np.arccos(rho / r0) - (rho / r0) * np.sqrt(1.0 - (rho / r0) ** 2.0)
    )
    A = np.nan_to_num(A)

    # region where (rho < (eta*r0)):
    B = (2.0 * eta**2.0 / np.pi) * (
        np.arccos(rho / eta / r0)
        - (rho / eta / r0) * np.sqrt(1.0 - (rho / eta / r0) ** 2.0)
    )
    B = np.nan_to_num(B)

    # region where (rho < ((1.0-eta)*r0/2.0)):
    C1 = -2.0 * eta**2.0 * (rho < (1.0 - eta) * r0 / 2.0)

    # region where (rho <= ((1.0+eta)*r0/2.0)):
    phi = np.arccos((1.0 + eta**2.0 - (2.0 * rho / r0) ** 2) / 2.0 / eta)
    C2 = (
        2.0 * eta * np.sin(phi) / np.pi
        + (1.0 + eta**2.0) * phi / np.pi
        - 2.0 * eta**2.0
    )
    C2 = C2 - (2.0 * (1.0 - eta**2.0) / np.pi) * np.arctan(
        (1.0 + eta) * np.tan(phi / 2.0) / (1.0 - eta)
    )
    C2 = np.nan_to_num(C2)
    C2 = C2 * (rho <= ((1.0 + eta) * r0 / 2.0))

    # note that C1+C2 = C from the IBSM documentation

    if eta > 0.0:
        H = (A + B + C1 + C2) / (1.0 - eta**2.0)
    else:
        H = A
    return H


def circularApertureOTFwithDefocus(
    u: np.ndarray,
    v: np.ndarray,
    wvl: float,
    D: float,
    f: float,
    defocus: float
) -> np.ndarray:

    """Calculate MTF for an unobscured circular aperture with a defocus
    aberration.From "The frequency response of a defocused optical system"
    (Hopkins, 1955) Variable changes made to use angular spatial frequency and
    approximation of 1/(F/#) = sin(a). Contributed by Matthew Howard.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param wvl:
        wavelength (m)
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)
    :param defocus:
        focus error distance between in focus and out of focus planes (m). In
        other words, this is the distance between the geometric focus and the
        actual focus.

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)
    :NOTE:
        Code contributed by Matt Howard
    """
    rho = np.sqrt(u**2.0 + v**2.0)  # radial spatial frequency
    r0 = D / wvl  # diffraction limited cutoff spatial frequency (cy/rad)

    s = 2.0 * rho / r0
    w20 = (
        0.5 / (1.0 + 4.0 * (f / D) ** 2.0) * defocus
    )  # note that this is the OPD error at
    # the edge of the pupil.  w20/wavelength is a commonly used specification
    # (e.g. waves of defocus)
    alpha = 4 * np.pi / wvl * w20 * s
    beta = np.arccos(0.5 * s)

    if defocus:
        defocus_otf = 2 / (np.pi * alpha) * np.cos(alpha * 0.5 * s) * (
            beta * jn(1, alpha)
            + 1 / 2.0 * np.sin(2 * beta * (jn(1, alpha) - jn(3, alpha)))
            - 1 / 4.0 * np.sin(4 * beta * (jn(3, alpha) - jn(5, alpha)))
            + 1 / 6.0 * np.sin(6 * beta * (jn(5, alpha) - jn(7, alpha)))
        ) - 2 / (np.pi * alpha) * np.sin(alpha * 0.5 * s) * (
            np.sin(beta * (jn(0, alpha) - jn(2, alpha)))
            - 1 / 3.0 * np.sin(3 * beta * (jn(2, alpha) - jn(4, alpha)))
            + 1 / 5.0 * np.sin(5 * beta * (jn(4, alpha) - jn(6, alpha)))
            - 1 / 7.0 * np.sin(7 * beta * (jn(6, alpha) - jn(8, alpha)))
        )

        defocus_otf[rho == 0] = 1
    else:
        defocus_otf = 1 / np.pi * (2 * beta - np.sin(2 * beta))

    H = np.nan_to_num(defocus_otf)

    return H


def cteOTF(
    u: np.ndarray,
    v: np.ndarray,
    px: float,
    py: float,
    cteNx: float,
    cteNy: float,
    phasesN: int,
    cteEff: float,
    f: float
) -> np.ndarray:
    """IBSM Equation 3-39.  Blur due to charge transfer efficiency losses in a
    CCD array.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param px:
        detector center-to-center spacings (pitch) in the x direction (m)
    :param py:
        detector center-to-center spacings (pitch) in the y direction (m)
    :param cteNx:
        number of change transfers in the x direction (unitless)
    :param cteNy:
        number of change transfers in the y direction (unitless)
    :param phasesN:
        number of clock phases per transfer (unitless)
    :param beta:
        ratio of TDI clocking rate to image motion rate (unitless)
    :param cteEff:
        charge transfer efficiency (unitless)
    :param f:
        focal length (m)

    :return:
        H:
            cte OTF
    """

    # this OTF has the same form in the x and y directions so we'll define
    # a function to save us the trouble of doing this twice N is either cteNx
    # or cteNy and pu is the product of pitch and spatial frequency - either
    # v*py or u*px
    def cteOTF_xy(
                N: float,
                pu: np.ndarray,
                phasesN: int,
                cteEff: float,
                f: float
                ) -> np.ndarray:
        return np.exp(
            -1.0
            * phasesN
            * N
            * (1.0 - cteEff)
            * (1.0 - np.cos(2.0 * np.pi * pu / f))
        )

    H = cteOTF_xy(cteNx, px * u, phasesN, cteEff, f) * cteOTF_xy(
        cteNy, py * v, phasesN, cteEff, f
    )

    return H


def defocusOTF(
    u: np.ndarray,
    v: np.ndarray,
    D: float,
    wx: float,
    wy: float
) -> np.ndarray:
    """IBSM Equation 3-25.  Gaussian approximation for defocus on the optical
    axis. This function is retained for backward compatibility.  See
    circularApertureOTFwithDefocus for an exact solution.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param D:
        effective aperture diameter (m)
    :param wx:
        the 1/e blur spot radii in the x direction
    :param wy:
        the 1/e blur spot radii in the y direction

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.exp(
        (-np.pi**2.0 / 4.0) * (wx**2.0 * u**2.0 + wy**2.0 * v**2.0)
    )

    return H


def detectorOTF(
    u: np.ndarray,
    v: np.ndarray,
    wx: float,
    wy: float,
    f: float
) -> np.ndarray:
    """A simplified version of IBSM Equation 3-36.  Blur due to the spatial
    integrating effects of the detector size.  See detectorOTFwithAggregation
    if detector aggregation is desired (new for version 1).


    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param wx:
        the 1/e blur spot radii in the x direction
    :param wy:
        the 1/e blur spot radii in the y direction
    :param f:
        focal length (m)

    :return:
        H:
            detector OTF. WARNRING: output can be NaN if f is 0
    """

    H = np.sinc(wx * u / f) * np.sinc(wy * v / f)

    return H


def detectorOTFwithAggregation(
    u: np.ndarray,
    v: np.ndarray,
    wx: float,
    wy: float,
    px: float,
    py: float,
    f: float,
    N: int = 1
) -> np.ndarray:
    """Blur due to the spatial integrating effects of the detector size and
    aggregation. Contributed by Matt Howard.  Derivation verified by Ken
    Barnard.  Note: this function is particularly important for aggregating
    detectors with less than 100% fill factor (e.g. px > wx).

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param wx:
        the 1/e blur spot radii in the x direction
    :param wy:
        the 1/e blur spot radii in the y direction
    :param px:
        detector pitch in the x direction (m)
    :param py:
        detector pitch in the y direction (m)
    :param f:
        focal length (m)
    :param N:
        number of pixels to aggregate

    :return:
        H:
            detector OTF
    :NOTE:
        Code contributed by Matt Howard
    """

    agg_u = 0.0
    agg_v = 0.0
    for i in range(N):
        phi_u = (
            2.0 * np.pi * ((i * px * u / f) - ((N - 1.0) * px * u / 2.0 / f))
        )
        agg_u = agg_u + np.cos(phi_u)
        phi_v = (
            2.0 * np.pi * ((i * py * v / f) - ((N - 1.0) * py * v / 2.0 / f))
        )
        agg_v = agg_v + np.cos(phi_v)

    H = (agg_u * agg_v / N**2) * np.sinc(wx * u / f) * np.sinc(wy * v / f)

    return H


def diffusionOTF(
    u: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    ald: float,
    al0: float,
    f: float
) -> float:
    """IBSM Equation 3-40.  Blur due to the effects of minority carrier
    diffusion in a CCD sensor.  Included for completeness but this isn't a good
    description of modern detector structures.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param alpha:
        carrier spectral diffusion coefficient (m^-1); note that IBSM Table 3-4
        contains alpha values as a function of wavelength for silicon
    :param ald:
        depletion layer width (m)
    :param al0:
        diffusion length (m)
    :param f:
        focal length (m)

    :return:
        H:
            diffusion OTF
    """

    def diffusionOTF_params(al: float, alpha: np.ndarray, ald: float) -> float:
        return 1.0 - np.exp(-alpha * ald) / (1.0 + alpha * al)

    rho = np.sqrt(u**2 + v**2)

    alrho = np.sqrt((1.0 / al0**2 + (2.0 * np.pi * rho / f) ** 2) ** (-1))

    H = diffusionOTF_params(alrho, alpha, ald) / diffusionOTF_params(
        al0, alpha, ald
    )

    return H


def driftOTF(
    u: np.ndarray,
    v: np.ndarray,
    ax: float,
    ay: float
) -> np.ndarray:
    """IBSM Equation 3-29.  Blur due to constant angular line-of-sight motion
    during the integration time.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param ax:
        line-of-sight angular drift during one integration time in the x
        direction respectively (rad)
    :param ay:
        line-of-sight angular drift during one integration time in the y
        direction respectively (rad)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.sinc(ax * u) * np.sinc(ay * v)

    return H


def filterOTF(
    u: np.ndarray,
    v: np.ndarray,
    kernel: np.ndarray,
    ifov: float
) -> np.ndarray:
    """Returns the OTF of any filter applied to the image (e.g. a sharpening
    filter).


    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param kernel:
         the 2-D image sharpening kernel; note that
         the kernel is assumed to sum to one
    :param ifov:
        instantaneous field-of-view of a detector (radians)

    :return:
        H:
            optical transfer function of the filter at spatial frequencies u
            and v
    """
    # most filter kernels are only a few pixels wide so we'll use zero-padding
    # to make the OTF larger.  The exact size doesn't matter too much
    # because the result is interpolated
    N = 100  # array size for the transform

    # transform of the kernel
    xferfcn = np.abs(np.fft.fftshift(np.fft.fft2(kernel, [N, N])))

    nyquist = 0.5 / ifov

    # spatial frequency coordinates for the transformed filter
    urng = np.linspace(-nyquist, nyquist, xferfcn.shape[0])
    vrng = np.linspace(nyquist, -nyquist, xferfcn.shape[1])
    nu, nv = np.meshgrid(urng, vrng)

    # reshape everything to comply with the griddata interpolator requirements
    xferfcn = xferfcn.reshape(-1)
    nu = nu.reshape(-1)
    nv = nv.reshape(-1)

    # use this function to wrap spatial frequencies beyond Nyquist
    def wrapval(value: np.ndarray, nyquist: float) -> np.ndarray:
        return (value + nyquist) % (2 * nyquist) - nyquist

    # and interpolate up to the desired range
    H = interpolate.griddata(
        (nu, nv),
        xferfcn,
        (wrapval(u, nyquist), wrapval(v, nyquist)),
        method="linear",
        fill_value=0,
    )

    return H


def gaussianOTF(
    u: np.ndarray,
    v: np.ndarray,
    blurSizeX: float,
    blurSizeY: float
) -> np.ndarray:
    """A real-valued Gaussian OTF.  This is useful for modeling systems when
    you have some general idea of the width of the point-spread-function or
    perhaps the cutoff frequency.  The blur size is defined to be where the PSF
    falls to about .043 times it's peak value.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param blurSizeX:
        angular extent of the blur spot in image space (radians)
    :param blurSizeY:
        angular extent of the blur spot in image space (radians)

    :return:
        H:
            gaussian optical transfer function

    :NOTE:
        The cutoff frequencies (where the MTF falls to .043 cycles/radian)
        are the inverse of the blurSizes and the point spread function is
        therefore:
        psf(x,y) = (fxX*fcY)*exp(-pi((fxX*x)^2+(fcY*y)^2))
    """
    fcX = 1 / blurSizeX  # x-direction cutoff frequency
    fcY = 1 / blurSizeY  # y-direction cutoff frequency

    H = np.exp(-np.pi * ((u / fcX) ** 2 + (v / fcY) ** 2))

    return H


def jitterOTF(
    u: np.ndarray,
    v: np.ndarray,
    sx: float,
    sy: float
) -> np.ndarray:
    """IBSM Equation 3-28.  Blur due to random line-of-sight motion that occurs
    at high.

    frequency, i.e. many small random changes in line-of-sight during a single
    integration time. Note that there is an error in Equation 3-28 - pi should
    be squared in the exponent.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param sx:
        Root-mean-squared jitter amplitudes in the x direction respectively.
        (rad)
    :param sy:
        Root-mean-squared jitter amplitudes in the y direction respectively.
        (rad)
    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.exp(
        (-2.0 * np.pi**2.0) * (sx**2.0 * u**2.0 + sy**2.0 * v**2.0)
    )

    return H


def polychromaticTurbulenceOTF(
    u: np.ndarray,
    v: np.ndarray,
    wavelengths: np.ndarray,
    weights: np.ndarray,
    altitude: float,
    slantRange: float,
    D: float,
    haWindspeed: float,
    cn2at1m: float,
    intTime: float,
    aircraftSpeed: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a polychromatic turbulence MTF based on the Hufnagel-Valley
    turbulence profile and the pyBSM function "windspeedTurbulenceOTF", i.e.
    IBSM Eqn 3.9.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param wavelengths:
        wavelength array (m)
    :param weights:
        how contributions from each wavelength are weighted
    :param altitude:
        height of the aircraft above the ground (m)
    :param slantRange:
        line-of-sight range between the aircraft and target (target is assumed
        to be on the ground)
    :param D:
        effective aperture diameter (m)
    :param intTime:
        dwell (i.e. integration) time (seconds)
    :param aircraftSpeed:
        apparent atmospheric velocity (m/s); this can just be the windspeed at
        the sensor position if the sensor is stationary
    :param haWindspeed:
        the high altitude windspeed (m/s); used to calculate the turbulence
        profile
    :param cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at
        h = 1 m); used to calculate the turbulence profile

    :return:
        turbulenceOTF:
            turbulence OTF (unitless)
        r0band:
            the effective coherence diameter across the band (m)

    :raises:
        ZeroDivisionError:
            if slantRange is 0
        IndexError:
            if weights or altitude if empty or the lengths of weights
            or altitude are not equal
    """
    # calculate the Structure constant along the slant path
    (zPath, hPath) = altitudeAlongSlantPath(0.0, altitude, slantRange)
    cn2 = hufnagelValleyTurbulenceProfile(hPath, haWindspeed, cn2at1m)

    # calculate the coherence diameter over the band
    r0at1um = coherenceDiameter(1.0e-6, zPath, cn2)
    r0function = (
        lambda wav: r0at1um * wav ** (6.0 / 5.0) * (1e-6) ** (-6.0 / 5.0)
    )
    r0band = weightedByWavelength(wavelengths, weights, r0function)

    # calculate the turbulence OTF
    turbFunction = (lambda wavelengths: windspeedTurbulenceOTF(
        u, v, wavelengths, D, r0function(wavelengths), intTime, aircraftSpeed
    ))
    turbulenceOTF = weightedByWavelength(wavelengths, weights, turbFunction)

    return turbulenceOTF, r0band


def radialUserOTF(
    u: np.ndarray,
    v: np.ndarray,
    fname: str
) -> np.ndarray:
    """IBSM Section 3.2.6.  Import a user-defined, 1-dimensional radial OTF and
    interpolate it onto a 2-dimensional spatial frequency grid.  Per ISBM
    Table.

    3-3a, the OTF data are ASCII text, space delimited data.  Each line of text
    is formatted as - spatial_frequency OTF_real OTF_imaginary.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param fname:
        filename and path to the radial OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    radialData = np.genfromtxt(fname)
    radialSF = np.sqrt(
        u**2.0 + v**2.0
    )  # calculate radial spatial frequencies

    H = (
        np.interp(radialSF, radialData[:, 0], radialData[:, 1])
        + np.interp(radialSF, radialData[:, 0], radialData[:, 2]) * 1.0j
    )

    return H


def tdiOTF(
    uorv: np.ndarray,
    w: float,
    ntdi: float,
    phasesN: int,
    beta: float,
    f: float
) -> np.ndarray:
    """IBSM Equation 3-38.  Blur due to a mismatch between the time-delay-
    integration clocking rate and the image motion.


    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param w:
        detector size (width) in the TDI direction (m)
    :param ntdi:
        number of TDI stages (unitless)
    :param phasesN:
        number of clock phases per transfer (unitless)
    :param beta:
        ratio of TDI clocking rate to image motion rate (unitless)
    :param f:
        focal length (m)

    :return:
        H:
            tdi OTF
    """

    xx = (
        w * uorv / (f * beta)
    )  # this occurs twice, so we'll pull it out to simplify the
    # the code

    expsum = 0.0
    iind = np.arange(0, ntdi * phasesN)  # goes from 0 to tdiN*phasesN-1
    for ii in iind:
        expsum = expsum + np.exp(-2.0j * np.pi * xx * (beta - 1.0) * ii)
    H = np.sinc(xx) * expsum / (ntdi * phasesN)
    return H


def turbulenceOTF(
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    D: float,
    r0: float,
    alpha: float
) -> np.ndarray:
    """IBSM Equation 3-3.  The long or short exposure turbulence OTF.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param lambda0:
        wavelength (m)
    :param D:
        effective aperture diameter (m)
    :param r0:
        Fried's correlation diameter (m)
    :param alpha:
        long exposure (alpha = 0) or short exposure (alpha = 0.5)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    :WARNING:
            Output can be inf if D is 0.
            Output can be nan if lambda0 and alpha are 0.
    """
    rho = np.sqrt(u**2.0 + v**2.0)  # radial spatial frequency
    H = np.exp(
        -3.44
        * (lambda0 * rho / r0) ** (5.0 / 3.0)
        * (1 - alpha * (lambda0 * rho / D) ** (1.0 / 3.0))
    )
    return H


def userOTF2D(
    u: np.ndarray,
    v: np.ndarray,
    fname: str,
    nyquist: float
) -> np.ndarray:
    """IBSM Section 3.2.7.  Import an user-defined, 2-dimensional OTF and
    interpolate onto a 2-dimensional spatial frequency grid.  The OTF data is
    assumed to be stored as a 2D Numpy array (e.g. 'fname.npy'); this is easier
    than trying to resurrect the IBSM image file format.  Zero spatial
    frequency is taken to be at the center of the array.  All OTFs values
    extrapolate to zero outside of the domain of the imported OTF.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param fname:
        filename and path to the OTF data; must include the .npy extension
    :param nyquist:
        the Nyquist (i.e. maximum) frequency of the OFT file; support
        of the OTF is assumed to extend from -nyquist to nyquist (rad^-1)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    rawOTF = np.load(fname)

    # find the row and column space of the raw OTF data
    vspace = np.linspace(1, -1, rawOTF.shape[0]) * nyquist
    uspace = np.linspace(-1, 1, rawOTF.shape[1]) * nyquist
    ugrid, vgrid = np.meshgrid(uspace, vspace)

    # reshape the data to be acceptable input to scipy's interpolate.griddata
    # this apparently works but I wonder if there is a better way?
    rawOTF = rawOTF.reshape(-1)
    ugrid = ugrid.reshape(-1)
    vgrid = vgrid.reshape(-1)

    H = interpolate.griddata(
        (ugrid, vgrid), rawOTF, (u, v), method="linear", fill_value=0
    )

    return H


def wavefrontOTF(
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    pv: float,
    Lx: float,
    Ly: float
) -> np.ndarray:
    """IBSM Equation 3-31.  Blur due to small random wavefront errors in the
    pupil. Use with the caution that this function assumes a specific phase
    autocorrelation function.  Refer to the discussion on random phase screens
    in Goodman, "Statistical Optics" for a full explanation (this is also the
    source cited in the IBSM documentation). As an alternative, see
    wavefrontOTF2.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param lambda0:
        wavelength (m)
    :param pv:
        phase variance (rad^2) -- tip: write as (2*pi*waves of error)^2
        (pv is often defined at a specific wavelength (e.g. 633 nm), so scale
        appropriately)
    :param Lx:
        correlation lengths of the phase autocorrelation function; apparently,
        it is common to set Lx to the aperture diameter (m)
    :param Ly:
        correlation lengths of the phase autocorrelation function; apparently,
        it is common to set Ly to the aperture diameter (m)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    :WARNING:
        Output can be nan if lambda0 is 0.

    """
    autoc = np.exp(-(lambda0**2) * ((u / Lx) ** 2 + (v / Ly) ** 2))
    H = np.exp(-pv * (1 - autoc))

    return H


def wavefrontOTF2(
    u: np.ndarray,
    v: np.ndarray,
    cutoff: float,
    wrms: float
) -> np.ndarray:
    """MTF due to wavefront errors.  In an ideal imaging system, a spherical
    waves converge to form an image at the focus.  Wavefront errors represent a
    departures from this ideal that lead to degraded image quality.  This
    function is an alternative to wavefrontOTF.  For more details see the R.
    Shannon, "Handbook of Optics," Chapter 35, "Optical Specifications."
    Useful notes from the author: for most imaging systems, wrms falls between
    0.1 and 0.25 waves rms.  This MTF becomes progressively less accurate as
    wrms exceeds .18 waves.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param cutoff:
        spatial frequency cutoff due to diffraction, i.e. aperture
        diameter / wavelength (rad^-1)
    :param wrms:
        root mean square wavefront error (waves of error)


    :return:
        H:
            wavefront OTF
    """

    v = np.sqrt(u**2.0 + v**2.0) / cutoff

    H = 1.0 - ((wrms / 0.18) ** 2.0) * (1.0 - 4.0 * (v - 0.5) ** 2.0)

    return H


def windspeedTurbulenceOTF(
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    D: float,
    r0: float,
    td: float,
    vel: float
) -> np.ndarray:
    """IBSM Equation 3-9.  Turbulence OTF adjusted for windspeed and
    integration time.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param lambda0:
        wavelength (m)
    :param D:
        effective aperture diameter (m)
    :param r0:
        Fried's coherence diameter (m)
    :param td:
        dwell (i.e. integration) time (seconds)
    :param vel:
        apparent atmospheric velocity (m/s)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    :raises:
        ZeroDivisionError:
            if r0 is 0

    :WARNING:
        Output can be nan if is D is 0.
    """
    weight = np.exp(-vel * td / r0)
    H = weight * turbulenceOTF(u, v, lambda0, D, r0, 0.5) + (
        1 - weight
    ) * turbulenceOTF(u, v, lambda0, D, r0, 0.0)
    return H


def xandyUserOTF(
    u: np.ndarray,
    v: np.ndarray,
    fname: str
) -> np.ndarray:
    """USE xandyUserOTF2 INSTEAD!  The original pyBSM documentation contains an
    error. IBSM Equation 3-32.  Import user-defined, 1-dimensional x-direction
    and y-direction OTFs and interpolate them onto a 2-dimensional spatial
    frequency grid.  Per ISBM Table.
    3-3c, the OTF data are ASCII text, space delimited data.  (Note: There
    appears to be a typo in the IBSM documentation - Table 3-3c should
    represent the "x and y" case, not "x or y".)

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param fname:
        filename and path to the x and y OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    xandyData = np.genfromtxt(fname)

    Hx = (
        np.interp(np.abs(u), xandyData[:, 0], xandyData[:, 1])
        + np.interp(np.abs(u), xandyData[:, 0], xandyData[:, 2]) * 1.0j
    )

    Hy = (
        np.interp(np.abs(v), xandyData[:, 3], xandyData[:, 4])
        + np.interp(np.abs(v), xandyData[:, 3], xandyData[:, 5]) * 1.0j
    )

    H = Hx * Hy

    return H


def xandyUserOTF2(
    u: np.ndarray,
    v: np.ndarray,
    fname: str
) -> np.ndarray:
    """UPDATE to IBSM Equation 3-32.  Import user-defined x-direction and
    y-direction OTFs and interpolate them onto a 2-dimensional spatial
    frequency grid.

    Per ISBM Table 3-3c, the OTF data are ASCII text, space delimited data.
    (Note: There appears to be a typo in the IBSM documentation - Table 3-3c
    should represent the "x and y" case, not "x or y".).  In the original
    version, the 2D OTF is given as Hx*Hy, the result being that the off-axis
    OTF is lower than either Hx or Hy. The output is now given by the geometric
    mean.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param fname:
        filename and path to the x and y OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    xandyData = np.genfromtxt(fname)

    Hx = (
        np.interp(np.abs(u), xandyData[:, 0], xandyData[:, 1])
        + np.interp(np.abs(u), xandyData[:, 0], xandyData[:, 2]) * 1.0j
    )

    Hy = (
        np.interp(np.abs(v), xandyData[:, 3], xandyData[:, 4])
        + np.interp(np.abs(v), xandyData[:, 3], xandyData[:, 5]) * 1.0j
    )

    H = np.sqrt(Hx * Hy)
    return H


# ----------------------------- END OTF Models -------------------------------

def otf2psf(
    otf: np.ndarray,
    df: float,
    dxout: float
) -> np.ndarray:
    """Transform an optical transfer function into a point spread function
    (i.e., image space blur filter)


    :param otf:
        Complex optical transfer function (OTF)
    :param df:
        Sample spacing for the optical transfer function (radians^-1)
    :param dxout:
        desired sample spacing of the point spread function (radians);
        WARNING: dxout must be small enough to properly sample the blur
        kernel

    :return:
        psf:
            blur kernel

    :raises:
        IndexError:
            if otf is not a 2D array
        ZeroDivisionError:
            if df or dxout are 0

    """
    # transform the psf
    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(otf))))

    # determine image space sampling
    dxin = 1 / (otf.shape[0] * df)

    # resample to the desired sample size
    newx = max([1, int(psf.shape[1] * dxin / dxout)])
    newy = max([1, int(psf.shape[0] * dxin / dxout)])
    psf = cv2.resize(psf, (newx, newy)).astype(np.float64)

    # ensure that the psf sums to 1
    psf = psf / psf.sum()

    # crop function for the desired kernel size
    getmiddle = (lambda x, ksize: x[
        tuple(
            [
                slice(
                    int(np.floor(d / 2 - ksize / 2)),
                    int(np.ceil(d / 2 + ksize / 2)),
                )
                for d in x.shape
            ]
        )
    ])

    # find the support region of the blur kernel
    for ii in np.arange(10, np.min(otf.shape), 5):
        psfout = getmiddle(psf, ii)
        if (
            psfout.sum() > 0.95
        ):  # note the 0.95 is heuristic (but seems to work well)
            break

    # make up for cropped out portions of the psf
    psfout = psfout / psfout.sum()  # bug fix 3 April 2020

    return psfout


def weightedByWavelength(
    wavelengths: np.ndarray,
    weights: np.ndarray,
    myFunction: Callable
) -> np.ndarray:
    """Returns a wavelength weighted composite array based on myFunction


    :param wavelengths:
        array of wavelengths (m)
    :param weights:
        array of weights corresponding to the "wavelengths" array;
        weights are normalized within this function so that weights.sum()==1
    :param myFunction:
        a lambda function parameterized by wavelength; e.g.
        otfFunction = lambda wavelengths: pybsm.otf.functional.circularApertureOTF
        (uu,vv,wavelengths,D,eta)

    :return:
        weightedfcn:
            the weighted function

    :raises:
        IndexError:
            if wavelengths or weights is empty or length of weights
            and wavelengths are not equal

    :WARNING:
        Output can be nan if all weights are 0.
    """
    weights = weights / weights.sum()
    weightedfcn = weights[0] * myFunction(wavelengths[0])

    for wii in wavelengths[1:]:
        weightedfcn = weightedfcn + weights[wavelengths == wii] * myFunction(
            wii
        )

    return weightedfcn


def coherenceDiameter(
    lambda0: float,
    zPath: np.ndarray,
    cn2: np.ndarray
) -> float:
    """
    This is an improvement / replacement for IBSM Equation 3-5: calculation of
    Fried's coherence diameter (m) for spherical wave propagation.
    It is primarily used in calculation of
    the turbulence OTF.  This version comes from Schmidt, "Numerical Simulation
    of Optical Wave Propagation with Examples in Matlab" SPIE Press (2010). In
    turn, Schmidt references Sasiela, "Electromagnetic Wave Propagation in
    Turbulence: Evaluation of Application of Mellin Transforms" SPIE Press
    (2007).


    :param lambda0:
        wavelength (m); to save time evaluating extra integrals, calculate r0
        at 1e-6 m then multiply by lambda^6/5 to scale to other wavelengths
    :param zPatH:
        array of samples along the path from the target (zPath = 0) to the
        sensor (m) -- WARNING: trapz will FAIL if you give it a two-element path;
        use a long zPath array, even if cn2 is constant
    :param cn2:
        refractive index structure parameter values at the sample locations in
        zPath (m^(-2/3)); typically Cn2 is a function of height so, as an
        intermediate step, calculate heights at each point along
        zPath (see altitudeAlongSlantPath)

    :return:
        r0:
            correlation diameter (m) at wavelength lambda0

    :raises:
        ValueError:
            if zPath is empty
        ZeroDivisionError:
            if lambda0 is 0

    :WARNING:
        r0 can be infinite if zPath is one element or if cn2 is one element and 0.
    """
    # the path integral of the structure parameter term
    spIntegral = np.trapz(cn2 * (zPath / zPath.max()) ** (5.0 / 3.0), zPath)

    r0 = (spIntegral * 0.423 * (2 * np.pi / lambda0) ** 2) ** (-3.0 / 5.0)

    return r0


def hufnagelValleyTurbulenceProfile(
    h: np.ndarray,
    v: float,
    cn2at1m: float
) -> np.ndarray:
    """Replaces IBSM Equations 3-6 through 3-8.  The Hufnagel-Valley Turbulence
    profile (i.e. a profile of the refractive index structure parameter as a
    function of altitude).  I suggest the HV profile because it seems to be in
    more widespread use than the profiles listed in the IBSM documentation.
    This is purely a personal choice.  The HV equation comes from Roggemann et
    al., "Imaging Through Turbulence", CRC Press (1996).  The often quoted HV
    5/7 model is a special case where Cn2at1m = 1.7e-14 and v = 21.  HV 5/7
    should result in a 5 cm coherence diameter (r0) and 7 urad isoplanatic
    angle along a vertical slant path into space.

    :param h:
        height above ground level in (m)
    :param v:
        the high altitude windspeed (m/s)
    :param cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at
        h = 1 m)

    :return:
        cn2 :
            refractive index structure parameter as a function of height
            (m^(-2/3))
    """
    cn2 = (
        5.94e-53 * (v / 27.0) ** 2.0 * h**10.0 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + cn2at1m * np.exp(-h / 100.0)
    )

    return cn2


def objectDomainDefocusRadii(
    D: float,
    R: float,
    R0: float,
) -> float:
    """IBSM Equation 3-26.  Axial defocus blur spot radii in the object domain.

    :param D:
        effective aperture diameter (m)
    :param R:
        object range (m)
    :param R0:
        range at which the focus is set (m)

    :return:
        w :
            the 1/e blur spot radii (rad) in one direction

    """
    w = 0.62 * D * (1.0 / R - 1.0 / R0)
    return w


def darkCurrentFromDensity(
    jd: float,
    wx: float,
    wy: float
) -> float:
    """The dark current part of Equation 3-42.  Use this function to calculate
    the total number of electrons generated from dark current during an
    integration time given a dark current density.  It is useful to separate
    this out from 3-42 for noise source analysis purposes and because sometimes
    dark current is defined in another way.

    :param jd:
        dark current density (A/m^2)
    :param wx:
        the 1/e blur spot radii in the x direction
    :param wy:
        the 1/e blur spot radii in the y direction

    :return:
        jde :
            dark current electron rate (e-/s); for TDI systems, just multiply
            the result by the number of TDI stages
    """
    jde = (
        jd * wx * wy / qc
    )  # recall that qc is defined as charge of an electron
    return jde


def imageDomainDefocusRadii(
    D: float,
    dz: float,
    f: float
) -> float:
    """IBSM Equation 3-27.  Axial defocus blur spot radii in the image domain.

    :param D:
        effective aperture diameter (m)
    :param dz:
        axial defocus in the image domain (m)
    :param f:
        focal length (m)

    :return:
        w :
            the 1/e blur spot radii (rad) in one direction

    """
    w = 0.62 * D * dz / (f**2.0)
    return w


def sliceotf(
    otf: np.ndarray,
    ang: float
) -> np.ndarray:
    """Returns a one dimensional slice of a 2D OTF (or MTF) along the direction
    specified by the input angle.

    :param otf:
        OTF defined by spatial frequencies (u,v) (unitless)
    :param ang:
        slice angle (radians); a 0 radian slice is along the u axis.  The
        angle rotates counterclockwise. Angle pi/2 is along the v axis.
    :return:
        oslice:
            one-dimensional OTF in the direction of angle; the sample spacing
            of oslice is the same as the original otf
    """
    u = np.linspace(-1.0, 1.0, otf.shape[0])
    v = np.linspace(1.0, -1.0, otf.shape[1])
    r = np.arange(0.0, 1.0, u[1] - u[0])

    f = interpolate.interp2d(u, v, otf)
    oslice = np.diag(f(r * np.cos(ang), r * np.sin(ang)))
    # the interpolator, f, calculates a bunch of points that we don't really
    # need since everything but the diagonal is thrown away.  It works but
    # it's inefficient.

    return oslice


def apply_otf_to_image(
    ref_img: np.ndarray,
    ref_gsd: float,
    ref_range: float,
    otf: np.ndarray,
    df: float,
    ifov: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies OTF to ideal reference image to simulate real imaging.

    We assume that 'ref_img' is an ideal, high-resolution view of the world
    (with known at-surface spatial resolution 'ref_gsd' in meters) that we
    would like to emulate what it looks like from some virtual camera. This
    virtual camera views the world along the same line of sight as the real
    camera that collected 'ref_img' but possibly at a different range
    'ref_range' and with a different (reduced) spatial resolution defined by
    'otf', 'df', and 'ifov'.

    The geometric assumptions employed by this process fall into several
    regimes of accuracy:

    (1) If our modeled virtual camera's assumed distance 'ref_range' matches
    that of the camera that actually acquired 'ref_img', then our geometric
    assumptions of this process hold up perfectly.

    (2) In cases where we are modeling a virtual camera at a different range,
    if the ranges are both large relative to the depth variation of the scene,
    then our approximations hold well. For remote sensing, particularly
    satellite imagery, this is a very good assumption because the depth
    variation of the world surface is inconsequential in comparison to the
    camera distance, which could be hundreds to thousands of kilometers away.

    (3) The remaining cases, such as ground-level imagery, where the scene
    depth variation could be sizable relative to the camera range, changing
    'ref_range' from that of the camera that actually captured 'ref_img' will
    result in unmodeled changes in perspective distortion. For example, you
    might have a foreground to your image, your object-of-interest at a mid-
    range, and then a background that goes off to the horizon. The best thing
    to do in this case is set 'ref_gsd' to that on the object-of-interest
    (since GSD will be different in the foreground and background) and
    interpret 'ref_range' is the distance from the virtual camera to the
    object-of-interest.

    :param ref_img:
        an ideal image of a view of the world that we want to emulate what it
        would look like from the imaging system defined by the remaining
        parameters
    :param ref_gsd:
        spatial sampling for 'ref_img' in meters; each pixel in 'ref_img' is
        assumed to capture a 'ref_gsd' x 'ref_gsd' square of some world
        surface. We assume the sampling is isotropic (x and y sampling are
        identical) and uniform across the whole field of view. This is
        generally a valid assumption for remote sensing imagery.
    :param ref_range:
        the assumed line of sight range from the virtual camera being simulated
        to the world surface or object-of-interest within 'ref_img'
    :param otf:
        the complex optical transfer function (OTF) of the imaging system as
        returned by the functions of pybsm.otf
    :param df:
        the spatial frequency sampling associated with 'otf' (radians^-1)
    :param ifov:
        instantaneous field of view (iFOV) of the virtual imaging system that
        we are modeling (radians)

    :return:
        sim_img:
            the blurred and resampled image
        sim_psf :
            the resampled blur kernel (useful for checking the health of the
            simulation)

    :raises:
        ZeroDivisionError:
            if ref_range is 0 or ifov is 0
        IndexError:
            if ref_img or otf are not 2D arrays

    :WARNING:
        ref_gsd *must* be small enough to properly sample the blur kernel. As a
        guide, if the image system transfer function goes to zero at angular
        spatial frequency, coff, then the sampling requirement will be readily
        met if ref_gsd <= ref_range/(4*coff).
    """

    # Generate a blur function from the OTF that is resampled to match the
    # angular dimensions of ref_img. We don't need to know the actual range
    # from the world surface and the camera that captured 'ref_img', but we do
    # know its spatial sampling distance (physical surface size captured by
    # each pixel). So, we imagine that the actual camera that collected
    # 'ref_img' was at a range of 'ref_range', same as the virtual camera we
    # are modeling. There are caveats to this discussed in the docstring for
    # this function. Therefore, we can calculate the instantaneous field of view
    # (iFOV) of the assumed real camera, which is
    # 2*arctan(ref_gsd/2/ref_range).
    psf = otf2psf(otf, df, 2 * np.arctan(ref_gsd / 2 / ref_range))

    # filter the image
    blurimg = cv2.filter2D(ref_img, -1, psf)

    # resample the image to the camera's ifov
    sim_img = resample2D(blurimg, ref_gsd / ref_range, ifov)

    # resample psf (good for health checks on the simulation)
    sim_psf = resample2D(psf, ref_gsd / ref_range, ifov)

    return sim_img, sim_psf


def commonOTFs(
    sensor: Sensor,
    scenario: Scenario,
    uu: np.ndarray,
    vv: np.ndarray,
    mtfwavelengths: np.ndarray,
    mtfweights: np.ndarray,
    slantRange: float,
    intTime: float
) -> OTF:
    """Returns optical transfer functions for the most common sources.  This
    code originally served the NIIRS model but has been abstracted for other
    uses. OTFs for the aperture, detector, turbulence, jitter, drift, wavefront
    errors, and image filtering are all explicitly considered.

    :param sensor:
        an object from the class sensor
    :param scenario:
        an object from the class scenario
    :param uu:
        spatial frequency arrays in the x directions respectively
        (cycles/radian)
    :param vv:
        spatial frequency arrays in the y directions respectively
        (cycles/radian)
    :param mtfwavelengths:
        a numpy array of wavelengths (m)
    :param mtfweights:
        a numpy array of weights for each wavelength contribution (arb)
    :param slantRange:
        distance between the sensor and the target (m)
    :param intTime:
        integration time (s)

    :return:
        otf:
            an object containing results of the OTF calculations along with
            many intermediate calculations; the full system OTF is contained
            in otf.systemOTF

    :raises:
        ZeroDivisionError:
            if slantRange is 0
        IndexError:
            if uu or vv are empty
    """

    otf = OTF()

    # aperture OTF
    apFunction = (lambda wavelengths: circularApertureOTF(
        uu, vv, wavelengths, sensor.D, sensor.eta
    ))
    otf.apOTF = weightedByWavelength(mtfwavelengths, mtfweights, apFunction)

    # turbulence OTF
    if (
        scenario.cn2at1m > 0.0
    ):  # this option allows you to turn off turbulence completely
        # by setting cn2 at the ground level to 0
        otf.turbOTF, otf.r0band = polychromaticTurbulenceOTF(
            uu,
            vv,
            mtfwavelengths,
            mtfweights,
            scenario.altitude,
            slantRange,
            sensor.D,
            scenario.haWindspeed,
            scenario.cn2at1m,
            intTime * sensor.ntdi,
            scenario.aircraftSpeed,
        )
    else:
        otf.turbOTF = np.ones(uu.shape)
        otf.r0band = 1e6 * np.ones(uu.shape)

    # detector OTF
    otf.detOTF = detectorOTF(uu, vv, sensor.wx, sensor.wy, sensor.f)

    # jitter OTF
    otf.jitOTF = jitterOTF(uu, vv, sensor.sx, sensor.sy)

    # drift OTF
    otf.drftOTF = driftOTF(
        uu,
        vv,
        sensor.dax * intTime * sensor.ntdi,
        sensor.day * intTime * sensor.ntdi,
    )

    # wavefront OTF
    wavFunction = (lambda wavelengths: wavefrontOTF(
        uu,
        vv,
        wavelengths,
        sensor.pv * (sensor.pvwavelength / wavelengths) ** 2,
        sensor.Lx,
        sensor.Ly,
    ))
    otf.wavOTF = weightedByWavelength(mtfwavelengths, mtfweights, wavFunction)

    # filter OTF (e.g. a sharpening filter but it could be anything)
    if sensor.filterKernel.shape[0] > 1:
        # note that we're assuming equal ifovs in the x and y directions
        otf.filterOTF = filterOTF(
            uu, vv, sensor.filterKernel, sensor.px / sensor.f
        )
    else:
        otf.filterOTF = np.ones(uu.shape)

    # system OTF
    otf.systemOTF = (
        otf.apOTF
        * otf.turbOTF
        * otf.detOTF
        * otf.jitOTF
        * otf.drftOTF
        * otf.wavOTF
        * otf.filterOTF
    )

    return otf


def resample2D(
    imgin: np.ndarray,
    dxin: float,
    dxout: float
) -> np.ndarray:
    """Resample an image.

    :param img:
        the input image
    :param dxin:
        sample spacing of the input image (radians)
    :param dxout:
        sample spacing of the output image (radians)

    :return:
        imgout:
            output image

    :raises:
        IndexError:
            if imigin is not a 2D array
        ZeroDivisionError:
            if dxout is 0
        cv2.error:
            if dxin is 0

    """

    newx = int(np.round(imgin.shape[1] * dxin / dxout))
    newy = int(np.round(imgin.shape[0] * dxin / dxout))
    imgout = cv2.resize(imgin, (newx, newy))

    return imgout
