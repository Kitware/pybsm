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
import inspect
import os
import warnings
from typing import Callable, Tuple

import cv2

# 3rd party imports
import numpy as np
from scipy import interpolate
from scipy.special import jn

# local imports
from pybsm.geospatial import altitude_along_slant_path
from pybsm.simulation.scenario import Scenario
from pybsm.simulation.sensor import Sensor

from .otf import OTF

# new in version 0.2.  We filter warnings associated with calculations in the
# function circular_aperture_OTF.  These invalid values are caught as NaNs and
# appropriately replaced.
warnings.filterwarnings("ignore", r"invalid value encountered in arccos")
warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")
warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
warnings.filterwarnings("ignore", r"divide by zero encountered in true_divide")

# find the current path (used to locate the atmosphere database)
# dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

# define some useful physical constants
hc = 6.62607004e-34  # Plank's constant  (m^2 kg / s)
cc = 299792458.0  # speed of light (m/s)
kc = 1.38064852e-23  # Boltzmann constant (m^2 kg / s^2 K)
qc = 1.60217662e-19  # charge of an electron (coulombs)
r_earth = 6378.164e3  # radius of the earth (m)


# ------------------------------- OTF Models ---------------------------------


def circular_aperture_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, lambda0: float, D: float, eta: float  # noqa: N803
) -> np.ndarray:
    """IBSM Equation 3-20.  Obscured circular aperture diffraction OTF.

    If eta is set to 0, the function will return the unobscured aperture result.

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
    a = (2.0 / np.pi) * (  # noqa: N806
        np.arccos(rho / r0) - (rho / r0) * np.sqrt(1.0 - (rho / r0) ** 2.0)
    )
    a = np.nan_to_num(a)  # noqa: N806

    # region where (rho < (eta*r0)):
    b = (2.0 * eta**2.0 / np.pi) * (  # noqa: N806
        np.arccos(rho / eta / r0)
        - (rho / eta / r0) * np.sqrt(1.0 - (rho / eta / r0) ** 2.0)
    )
    b = np.nan_to_num(b)  # noqa: N806

    # region where (rho < ((1.0-eta)*r0/2.0)):
    c_1 = -2.0 * eta**2.0 * (rho < (1.0 - eta) * r0 / 2.0)  # noqa: N806

    # region where (rho <= ((1.0+eta)*r0/2.0)):
    phi = np.arccos((1.0 + eta**2.0 - (2.0 * rho / r0) ** 2) / 2.0 / eta)
    c_2 = (  # noqa: N806
        2.0 * eta * np.sin(phi) / np.pi
        + (1.0 + eta**2.0) * phi / np.pi
        - 2.0 * eta**2.0
    )
    c_2 = c_2 - (2.0 * (1.0 - eta**2.0) / np.pi) * np.arctan(  # noqa: N806
        (1.0 + eta) * np.tan(phi / 2.0) / (1.0 - eta)
    )
    c_2 = np.nan_to_num(c_2)  # noqa: N806
    c_2 = c_2 * (rho <= ((1.0 + eta) * r0 / 2.0))  # noqa: N806

    # note that c_1+c_2 = C from the IBSM documentation

    if eta > 0.0:
        H = (a + b + c_1 + c_2) / (1.0 - eta**2.0)  # noqa: N806
    else:
        H = a  # noqa: N806
    return H


def circular_aperture_OTF_with_defocus(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    wavelength: float,
    D: float,  # noqa: N803
    f: float,
    defocus: float,
) -> np.ndarray:
    """Calculate MTF for an unobscured circular aperture with a defocus aberration.

    From "The frequency response of a defocused optical system" (Hopkins, 1955)
    Variable changes made to use angular spatial frequency and
    approximation of 1/(F/#) = sin(a). Contributed by Matthew Howard.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param wavelength:
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
    r0 = D / wavelength  # diffraction limited cutoff spatial frequency (cy/rad)

    s = 2.0 * rho / r0
    w20 = (
        0.5 / (1.0 + 4.0 * (f / D) ** 2.0) * defocus
    )  # note that this is the OPD error at
    # the edge of the pupil.  w20/wavelength is a commonly used specification
    # (e.g. waves of defocus)
    alpha = 4 * np.pi / wavelength * w20 * s
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

    H = np.nan_to_num(defocus_otf)  # noqa: N806

    return H


def cte_OTF(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    p_x: float,
    p_y: float,
    cte_n_x: float,
    cte_n_y: float,
    phases_n: int,
    cte_eff: float,
    f: float,
) -> np.ndarray:
    """IBSM Equation 3-39.  Blur due to charge transfer efficiency losses in a CCD array.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param p_x:
        detector center-to-center spacings (pitch) in the x direction (m)
    :param p_y:
        detector center-to-center spacings (pitch) in the y direction (m)
    :param cte_n_x:
        number of change transfers in the x direction (unitless)
    :param cte_n_y:
        number of change transfers in the y direction (unitless)
    :param phases_n:
        number of clock phases per transfer (unitless)
    :param beta:
        ratio of TDI clocking rate to image motion rate (unitless)
    :param cte_eff:
        charge transfer efficiency (unitless)
    :param f:
        focal length (m)

    :return:
        H:
            cte OTF
    """

    # this OTF has the same form in the x and y directions so we'll define
    # a function to save us the trouble of doing this twice n is either cte_n_x
    # or cte_n_y and pu is the product of pitch and spatial frequency - either
    # v*p_y or u*p_x
    def cte_OTF_xy(  # noqa: N802
        n: float, pu: np.ndarray, phases_n: int, cte_eff: float, f: float
    ) -> np.ndarray:
        return np.exp(
            -1.0 * phases_n * n * (1.0 - cte_eff) * (1.0 - np.cos(2.0 * np.pi * pu / f))
        )

    H = cte_OTF_xy(cte_n_x, p_x * u, phases_n, cte_eff, f) * cte_OTF_xy(  # noqa: N806
        cte_n_y, p_y * v, phases_n, cte_eff, f
    )

    return H


def defocus_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, D: float, w_x: float, w_y: float  # noqa: N803
) -> np.ndarray:
    """IBSM Equation 3-25.  Gaussian approximation for defocus on the optical axis.

    This function is retained for backward compatibility.  See circular_aperture_OTF_with_defocus for an exact solution.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param D:
        effective aperture diameter (m)
    :param w_x:
        the 1/e blur spot radii in the x direction
    :param w_y:
        the 1/e blur spot radii in the y direction

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.exp(  # noqa: N806
        (-np.pi**2.0 / 4.0) * (w_x**2.0 * u**2.0 + w_y**2.0 * v**2.0)
    )

    return H


def detector_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, w_x: float, w_y: float, f: float
) -> np.ndarray:
    """A simplified version of IBSM Equation 3-36.

    Blur due to the spatial integrating effects of the detector size.  See detector_OTF_with_aggregation
    if detector aggregation is desired (new for version 1).


    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param w_x:
        the 1/e blur spot radii in the x direction
    :param w_y:
        the 1/e blur spot radii in the y direction
    :param f:
        focal length (m)

    :return:
        H:
            detector OTF. WARNRING: output can be NaN if f is 0
    """
    H = np.sinc(w_x * u / f) * np.sinc(w_y * v / f)  # noqa: N806

    return H


def detector_OTF_with_aggregation(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    w_x: float,
    w_y: float,
    p_x: float,
    p_y: float,
    f: float,
    n: int = 1,
) -> np.ndarray:
    """Blur due to the spatial integrating effects of the detector size and aggregation.

    Contributed by Matt Howard.  Derivation verified by Ken Barnard.
    Note: this function is particularly important for aggregating
    detectors with less than 100% fill factor (e.g. p_x > w_x).

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param w_x:
        the 1/e blur spot radii in the x direction
    :param w_y:
        the 1/e blur spot radii in the y direction
    :param p_x:
        detector pitch in the x direction (m)
    :param p_y:
        detector pitch in the y direction (m)
    :param f:
        focal length (m)
    :param n:
        number of pixels to aggregate

    :return:
        H:
            detector OTF
    :NOTE:
        Code contributed by Matt Howard
    """
    agg_u = 0.0
    agg_v = 0.0
    for i in range(n):
        phi_u = 2.0 * np.pi * ((i * p_x * u / f) - ((n - 1.0) * p_x * u / 2.0 / f))
        agg_u = agg_u + np.cos(phi_u)
        phi_v = 2.0 * np.pi * ((i * p_y * v / f) - ((n - 1.0) * p_y * v / 2.0 / f))
        agg_v = agg_v + np.cos(phi_v)

    H = (  # noqa: N806
        (agg_u * agg_v / n**2) * np.sinc(w_x * u / f) * np.sinc(w_y * v / f)
    )

    return H


def diffusion_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, alpha: np.ndarray, ald: float, al0: float, f: float
) -> float:
    """IBSM Equation 3-40.  Blur due to the effects of minority carrier diffusion in a CCD sensor.

    Included for completeness but this isn't a good description of modern detector structures.

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

    def diffusion_OTF_params(  # noqa: N802
        al: float, alpha: np.ndarray, ald: float
    ) -> float:
        return 1.0 - np.exp(-alpha * ald) / (1.0 + alpha * al)

    rho = np.sqrt(u**2 + v**2)

    al_rho = np.sqrt((1.0 / al0**2 + (2.0 * np.pi * rho / f) ** 2) ** (-1))

    H = diffusion_OTF_params(al_rho, alpha, ald) / diffusion_OTF_params(  # noqa: N806
        al0, alpha, ald
    )

    return H


def drift_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, a_x: float, a_y: float
) -> np.ndarray:
    """IBSM Equation 3-29.  Blur due to constant angular line-of-sight motion during the integration time.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param a_x:
        line-of-sight angular drift during one integration time in the x
        direction respectively (rad)
    :param a_y:
        line-of-sight angular drift during one integration time in the y
        direction respectively (rad)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.sinc(a_x * u) * np.sinc(a_y * v)  # noqa: N806

    return H


def filter_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, kernel: np.ndarray, ifov: float
) -> np.ndarray:
    """Returns the OTF of any filter applied to the image (e.g. a sharpening filter).

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
    n = 100  # array size for the transform

    # transform of the kernel
    xfer_fcn = np.abs(np.fft.fftshift(np.fft.fft2(kernel, [n, n])))

    nyquist = 0.5 / ifov

    # spatial frequency coordinates for the transformed filter
    u_rng = np.linspace(-nyquist, nyquist, xfer_fcn.shape[0])
    v_rng = np.linspace(nyquist, -nyquist, xfer_fcn.shape[1])
    n_u, n_v = np.meshgrid(u_rng, v_rng)

    # reshape everything to comply with the griddata interpolator requirements
    xfer_fcn = xfer_fcn.reshape(-1)
    n_u = n_u.reshape(-1)
    n_v = n_v.reshape(-1)

    # use this function to wrap spatial frequencies beyond Nyquist
    def wrap_val(value: np.ndarray, nyquist: float) -> np.ndarray:
        return (value + nyquist) % (2 * nyquist) - nyquist

    # and interpolate up to the desired range
    H = interpolate.griddata(  # noqa: N806
        (n_u, n_v),
        xfer_fcn,
        (wrap_val(u, nyquist), wrap_val(v, nyquist)),
        method="linear",
        fill_value=0,
    )

    return H


def gaussian_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, blur_size_x: float, blur_size_y: float
) -> np.ndarray:
    """A real-valued Gaussian OTF.

    This is useful for modeling systems when you have some general idea of the width of the point-spread-function or
    perhaps the cutoff frequency.  The blur size is defined to be where the PSF falls to about .043 times
    it's peak value.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param blur_size_x:
        angular extent of the blur spot in image space (radians)
    :param blur_size_y:
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
    fcX = 1 / blur_size_x  # x-direction cutoff frequency    # noqa: N806
    fcY = 1 / blur_size_y  # y-direction cutoff frequency    # noqa: N806

    H = np.exp(-np.pi * ((u / fcX) ** 2 + (v / fcY) ** 2))  # noqa: N806

    return H


def jitter_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, s_x: float, s_y: float
) -> np.ndarray:
    """IBSM Equation 3-28.  Blur due to random line-of-sight motion that occurs at high.

    frequency, i.e. many small random changes in line-of-sight during a single
    integration time. Note that there is an error in Equation 3-28 - pi should
    be squared in the exponent.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param s_x:
        Root-mean-squared jitter amplitudes in the x direction respectively.
        (rad)
    :param s_y:
        Root-mean-squared jitter amplitudes in the y direction respectively.
        (rad)
    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    H = np.exp(  # noqa: N806
        (-2.0 * np.pi**2.0) * (s_x**2.0 * u**2.0 + s_y**2.0 * v**2.0)
    )

    return H


def polychromatic_turbulence_OTF(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    wavelengths: np.ndarray,
    weights: np.ndarray,
    altitude: float,
    slant_range: float,
    D: float,  # noqa: N803
    ha_wind_speed: float,
    cn2_at_1m: float,
    int_time: float,
    aircraft_speed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """IBSM Eqn 3.9. Returns a polychromatic turbulence MTF.

    Returns a polychromatic turbulence MTF based on the Hufnagel-Valley turbulence profile
    and the pyBSM function "wind_speed_turbulence_OTF", i.e. IBSM Eqn 3.9.

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
    :param slant_range:
        line-of-sight range between the aircraft and target (target is assumed
        to be on the ground)
    :param D:
        effective aperture diameter (m)
    :param int_time:
        dwell (i.e. integration) time (seconds)
    :param aircraft_speed:
        apparent atmospheric velocity (m/s); this can just be the windspeed at
        the sensor position if the sensor is stationary
    :param ha_wind_speed:
        the high altitude windspeed (m/s); used to calculate the turbulence
        profile
    :param cn2_at_1m:
        the refractive index structure parameter "near the ground" (e.g. at
        h = 1 m); used to calculate the turbulence profile

    :return:
        turbulence_OTF:
            turbulence OTF (unitless)
        r0_band:
            the effective coherence diameter across the band (m)

    :raises:
        ZeroDivisionError:
            if slant_range is 0
        IndexError:
            if weights or altitude if empty or the lengths of weights
            or altitude are not equal
    """
    # calculate the Structure constant along the slant path
    (z_path, h_path) = altitude_along_slant_path(0.0, altitude, slant_range)
    cn2 = hufnagel_valley_turbulence_profile(h_path, ha_wind_speed, cn2_at_1m)

    # calculate the coherence diameter over the band
    r0_at_1um = coherence_diameter(1.0e-6, z_path, cn2)
    r0_function = (
        lambda wav: r0_at_1um  # noqa: E731
        * wav ** (6.0 / 5.0)
        * (1e-6) ** (-6.0 / 5.0)
    )
    r0_band = weighted_by_wavelength(wavelengths, weights, r0_function)

    # calculate the turbulence OTF
    turb_function = lambda wavelengths: wind_speed_turbulence_OTF(  # noqa: E731
        u, v, wavelengths, D, r0_function(wavelengths), int_time, aircraft_speed
    )
    turbulence_OTF = weighted_by_wavelength(  # noqa: N806
        wavelengths, weights, turb_function
    )

    return turbulence_OTF, r0_band


def radial_user_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, f_name: str
) -> np.ndarray:
    """IBSM Section 3.2.6.

    Import a user-defined, 1-dimensional radial OTF and interpolate it onto a 2-dimensional spatial frequency grid.
    Per ISBM Table.

    3-3a, the OTF data are ASCII text, space delimited data.  Each line of text
    is formatted as - spatial_frequency OTF_real OTF_imaginary.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param f_name:
        filename and path to the radial OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    radial_data = np.genfromtxt(f_name)
    radial_sf = np.sqrt(u**2.0 + v**2.0)  # calculate radial spatial frequencies

    H = (  # noqa: N806
        np.interp(radial_sf, radial_data[:, 0], radial_data[:, 1])
        + np.interp(radial_sf, radial_data[:, 0], radial_data[:, 2]) * 1.0j
    )

    return H


def tdi_OTF(  # noqa: N802
    u_or_v: np.ndarray, w: float, n_tdi: float, phases_n: int, beta: float, f: float
) -> np.ndarray:
    """IBSM Equation 3-38.  Blur due to a mismatch between the time-delay-integration clocking rate and image motion.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param w:
        detector size (width) in the TDI direction (m)
    :param n_tdi:
        number of TDI stages (unitless)
    :param phases_n:
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
        w * u_or_v / (f * beta)
    )  # this occurs twice, so we'll pull it out to simplify the
    # the code

    exp_sum = 0.0
    iind = np.arange(0, n_tdi * phases_n)  # goes from 0 to tdiN*phases_n-1
    for ii in iind:
        exp_sum = exp_sum + np.exp(-2.0j * np.pi * xx * (beta - 1.0) * ii)
    H = np.sinc(xx) * exp_sum / (n_tdi * phases_n)  # noqa: N806
    return H


def turbulence_OTF(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    D: float,  # noqa: N803
    r0: float,
    alpha: float,
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
    H = np.exp(  # noqa: N806
        -3.44
        * (lambda0 * rho / r0) ** (5.0 / 3.0)
        * (1 - alpha * (lambda0 * rho / D) ** (1.0 / 3.0))
    )
    return H


def user_OTF_2D(  # noqa: N802
    u: np.ndarray, v: np.ndarray, f_name: str, nyquist: float
) -> np.ndarray:
    """IBSM Section 3.2.7.  Import an user-defined, 2D OTF and interpolate onto a 2D spatial frequency grid.

    The OTF data is assumed to be stored as a 2D Numpy array (e.g. 'f_name.npy'); this is easier
    than trying to resurrect the IBSM image file format.  Zero spatial frequency is taken to be at the
    center of the array.  All OTFs values extrapolate to zero outside of the domain of the imported OTF.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param f_name:
        filename and path to the OTF data; must include the .npy extension
    :param nyquist:
        the Nyquist (i.e. maximum) frequency of the OFT file; support
        of the OTF is assumed to extend from -nyquist to nyquist (rad^-1)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    raw_OTF = np.load(f_name)  # noqa: N806

    # find the row and column space of the raw OTF data
    v_space = np.linspace(1, -1, raw_OTF.shape[0]) * nyquist
    u_space = np.linspace(-1, 1, raw_OTF.shape[1]) * nyquist
    u_grid, v_grid = np.meshgrid(u_space, v_space)

    # reshape the data to be acceptable input to scipy's interpolate.griddata
    # this apparently works but I wonder if there is a better way?
    raw_OTF = raw_OTF.reshape(-1)  # noqa: N806
    u_grid = u_grid.reshape(-1)
    v_grid = v_grid.reshape(-1)

    H = interpolate.griddata(  # noqa: N806
        (u_grid, v_grid), raw_OTF, (u, v), method="linear", fill_value=0
    )

    return H


def wavefront_OTF(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    pv: float,
    L_x: float,  # noqa: N803
    L_y: float,  # noqa: N803
) -> np.ndarray:
    """IBSM Equation 3-31.  Blur due to small random wavefront errors in the pupil.

    Use with the caution that this function assumes a specific phase
    autocorrelation function.  Refer to the discussion on random phase screens
    in Goodman, "Statistical Optics" for a full explanation (this is also the
    source cited in the IBSM documentation). As an alternative, see
    wavefront_OTF_2.

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
    :param L_x:
        correlation lengths of the phase autocorrelation function; apparently,
        it is common to set L_x to the aperture diameter (m)
    :param L_y:
        correlation lengths of the phase autocorrelation function; apparently,
        it is common to set L_y to the aperture diameter (m)

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    :WARNING:
        Output can be nan if lambda0 is 0.

    """
    auto_c = np.exp(-(lambda0**2) * ((u / L_x) ** 2 + (v / L_y) ** 2))
    H = np.exp(-pv * (1 - auto_c))  # noqa: N806

    return H


def wavefront_OTF_2(  # noqa: N802
    u: np.ndarray, v: np.ndarray, cutoff: float, w_rms: float
) -> np.ndarray:
    """This function is an alternative to wavefront_OTF. MTF due to wavefront errors.

    In an ideal imaging system, a spherical waves converge to form an image at the focus.
    Wavefront errors represent a departures from this ideal that lead to degraded image quality.
    This function is an alternative to wavefront_OTF.  For more details see the R.
    Shannon, "Handbook of Optics," Chapter 35, "Optical Specifications."
    Useful notes from the author: for most imaging systems, w_rms falls between
    0.1 and 0.25 waves rms.  This MTF becomes progressively less accurate as
    w_rms exceeds .18 waves.

    :param u:
        spatial frequency coordinates (rad^-1)
    :param v:
        spatial frequency coordinates (rad^-1)
    :param cutoff:
        spatial frequency cutoff due to diffraction, i.e. aperture
        diameter / wavelength (rad^-1)
    :param w_rms:
        root mean square wavefront error (waves of error)


    :return:
        H:
            wavefront OTF
    """
    v = np.sqrt(u**2.0 + v**2.0) / cutoff

    H = 1.0 - ((w_rms / 0.18) ** 2.0) * (1.0 - 4.0 * (v - 0.5) ** 2.0)  # noqa: N806

    return H


def wind_speed_turbulence_OTF(  # noqa: N802
    u: np.ndarray,
    v: np.ndarray,
    lambda0: float,
    D: float,  # noqa: N803
    r0: float,
    t_d: float,
    vel: float,
) -> np.ndarray:
    """IBSM Equation 3-9.  Turbulence OTF adjusted for windspeed and integration time.

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
    :param t_d:
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
    weight = np.exp(-vel * t_d / r0)
    H = weight * turbulence_OTF(u, v, lambda0, D, r0, 0.5) + (  # noqa: N806
        1 - weight
    ) * turbulence_OTF(u, v, lambda0, D, r0, 0.0)
    return H


def x_and_y_user_OTF(  # noqa: N802
    u: np.ndarray, v: np.ndarray, f_name: str
) -> np.ndarray:
    """USE x_and_y_user_OTF_2 INSTEAD!  The original pyBSM documentation contains an error.

    IBSM Equation 3-32. Import user-defined, 1-dimensional x-direction and y-direction OTFs and interpolate them onto
    a 2-dimensional spatial frequency grid.  Per ISBM Table. 3-3c, the OTF data are ASCII text, space delimited data.
    (Note: There appears to be a typo in the IBSM documentation - Table 3-3c should represent the "x and y" case,
    not "x or y".)

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param f_name:
        filename and path to the x and y OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    x_and_y_data = np.genfromtxt(f_name)

    H_x = (  # noqa: N806
        np.interp(np.abs(u), x_and_y_data[:, 0], x_and_y_data[:, 1])
        + np.interp(np.abs(u), x_and_y_data[:, 0], x_and_y_data[:, 2]) * 1.0j
    )

    H_y = (  # noqa: N806
        np.interp(np.abs(v), x_and_y_data[:, 3], x_and_y_data[:, 4])
        + np.interp(np.abs(v), x_and_y_data[:, 3], x_and_y_data[:, 5]) * 1.0j
    )

    H = H_x * H_y  # noqa: N806

    return H


def x_and_y_user_OTF_2(  # noqa: N802
    u: np.ndarray, v: np.ndarray, f_name: str
) -> np.ndarray:
    """UPDATE to IBSM Equation 3-32.

    Import user-defined x-direction and y-direction OTFs and interpolate them onto a 2D spatial frequency grid.

    Per ISBM Table 3-3c, the OTF data are ASCII text, space delimited data.
    (Note: There appears to be a typo in the IBSM documentation - Table 3-3c
    should represent the "x and y" case, not "x or y".).  In the original
    version, the 2D OTF is given as H_x*H_y, the result being that the off-axis
    OTF is lower than either H_x or H_y. The output is now given by the geometric
    mean.

    :param u:
        angular spatial frequency coordinates (rad^-1)
    :param v:
        angular spatial frequency coordinates (rad^-1)
    :param f_name:
        filename and path to the x and y OTF data

    :return:
        H:
            OTF at spatial frequency (u,v) (unitless)

    """
    x_and_y_data = np.genfromtxt(f_name)

    H_x = (  # noqa: N806
        np.interp(np.abs(u), x_and_y_data[:, 0], x_and_y_data[:, 1])
        + np.interp(np.abs(u), x_and_y_data[:, 0], x_and_y_data[:, 2]) * 1.0j
    )

    H_y = (  # noqa: N806
        np.interp(np.abs(v), x_and_y_data[:, 3], x_and_y_data[:, 4])
        + np.interp(np.abs(v), x_and_y_data[:, 3], x_and_y_data[:, 5]) * 1.0j
    )

    H = np.sqrt(H_x * H_y)  # noqa: N806
    return H


# ----------------------------- END OTF Models -------------------------------


def otf_to_psf(otf: np.ndarray, df: float, dx_out: float) -> np.ndarray:
    """Transform an optical transfer function into a point spread function (i.e., image space blur filter).

    :param otf:
        Complex optical transfer function (OTF)
    :param df:
        Sample spacing for the optical transfer function (radians^-1)
    :param dx_out:
        desired sample spacing of the point spread function (radians);
        WARNING: dx_out must be small enough to properly sample the blur
        kernel

    :return:
        psf:
            blur kernel

    :raises:
        IndexError:
            if otf is not a 2D array
        ZeroDivisionError:
            if df or dx_out are 0

    """
    # transform the psf
    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(otf))))

    # determine image space sampling
    dx_in = 1 / (otf.shape[0] * df)

    # resample to the desired sample size
    new_x = max([1, int(psf.shape[1] * dx_in / dx_out)])
    new_y = max([1, int(psf.shape[0] * dx_in / dx_out)])
    psf = cv2.resize(psf, (new_x, new_y)).astype(np.float64)

    # ensure that the psf sums to 1
    psf = psf / psf.sum()

    # crop function for the desired kernel size
    get_middle = lambda x, k_size: x[  # noqa: E731
        tuple(
            [
                slice(
                    int(np.floor(d / 2 - k_size / 2)),
                    int(np.ceil(d / 2 + k_size / 2)),
                )
                for d in x.shape
            ]
        )
    ]

    # find the support region of the blur kernel
    for ii in np.arange(10, np.min(otf.shape), 5):
        psf_out = get_middle(psf, ii)
        if psf_out.sum() > 0.95:  # note the 0.95 is heuristic (but seems to work well)
            break

    # make up for cropped out portions of the psf
    psf_out = psf_out / psf_out.sum()  # bug fix 3 April 2020

    return psf_out


def weighted_by_wavelength(
    wavelengths: np.ndarray, weights: np.ndarray, my_function: Callable
) -> np.ndarray:
    """Returns a wavelength weighted composite array based on my_function.

    :param wavelengths:
        array of wavelengths (m)
    :param weights:
        array of weights corresponding to the "wavelengths" array;
        weights are normalized within this function so that weights.sum()==1
    :param my_function:
        a lambda function parameterized by wavelength; e.g.
        otfFunction = lambda wavelengths: pybsm.otf.functional.circular_aperture_OTF
        (uu,vv,wavelengths,D,eta)

    :return:
        weighted_fcn:
            the weighted function

    :raises:
        IndexError:
            if wavelengths or weights is empty or length of weights
            and wavelengths are not equal

    :WARNING:
        Output can be nan if all weights are 0.
    """
    weights = weights / weights.sum()
    weighted_fcn = weights[0] * my_function(wavelengths[0])

    for wii in wavelengths[1:]:
        weighted_fcn = weighted_fcn + weights[wavelengths == wii] * my_function(wii)

    return weighted_fcn


def coherence_diameter(lambda0: float, z_path: np.ndarray, cn2: np.ndarray) -> float:
    """Improvement / replacement for calculation of Fried's coherence diameter (m) for spherical wave propagation.

    This is an improvement / replacement for IBSM Equation 3-5: calculation of Fried's coherence diameter (m) for
    spherical wave propagation. It is primarily used in calculation of the turbulence OTF.  This version comes from
    Schmidt, "Numerical Simulation of Optical Wave Propagation with Examples in Matlab" SPIE Press (2010). In
    turn, Schmidt references Sasiela, "Electromagnetic Wave Propagation in
    Turbulence: Evaluation of Application of Mellin Transforms" SPIE Press
    (2007).


    :param lambda0:
        wavelength (m); to save time evaluating extra integrals, calculate r0
        at 1e-6 m then multiply by lambda^6/5 to scale to other wavelengths
    :param z_path:
        array of samples along the path from the target (z_path = 0) to the
        sensor (m) -- WARNING: trapz will FAIL if you give it a two-element path;
        use a long z_path array, even if cn2 is constant
    :param cn2:
        refractive index structure parameter values at the sample locations in
        z_path (m^(-2/3)); typically Cn2 is a function of height so, as an
        intermediate step, calculate heights at each point along
        z_path (see altitude_along_slant_path)

    :return:
        r0:
            correlation diameter (m) at wavelength lambda0

    :raises:
        ValueError:
            if z_path is empty
        ZeroDivisionError:
            if lambda0 is 0

    :WARNING:
        r0 can be infinite if z_path is one element or if cn2 is one element and 0.
    """
    # the path integral of the structure parameter term
    sp_integral = np.trapz(cn2 * (z_path / z_path.max()) ** (5.0 / 3.0), z_path)

    r0 = (sp_integral * 0.423 * (2 * np.pi / lambda0) ** 2) ** (-3.0 / 5.0)

    return r0


def hufnagel_valley_turbulence_profile(
    h: np.ndarray, v: float, cn2_at_1m: float
) -> np.ndarray:
    """Replaces IBSM Equations 3-6 through 3-8.  The Hufnagel-Valley Turbulence profile.

    Replaces IBSM Equations 3-6 through 3-8.  The Hufnagel-Valley Turbulence profile (i.e. a
    profile of the refractive index structure parameter as a function of altitude).
    I suggest the HV profile because it seems to be in more widespread use than the profiles
    listed in the IBSM documentation. This is purely a personal choice.  The HV equation comes from Roggemann et
    al., "Imaging Through Turbulence", CRC Press (1996).  The often quoted HV
    5/7 model is a special case where Cn2at1m = 1.7e-14 and v = 21.  HV 5/7
    should result in a 5 cm coherence diameter (r0) and 7 urad isoplanatic
    angle along a vertical slant path into space.

    :param h:
        height above ground level in (m)
    :param v:
        the high altitude windspeed (m/s)
    :param cn2_at_1m:
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
        + cn2_at_1m * np.exp(-h / 100.0)
    )

    return cn2


def object_domain_defocus_radii(
    D: float,  # noqa: N803
    R: float,  # noqa: N803
    R0: float,  # noqa: N803
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


def dark_current_from_density(jd: float, w_x: float, w_y: float) -> float:
    """The dark current part of Equation 3-42.

    Use this function to calculate the total number of electrons generated from dark current during an
    integration time given a dark current density.  It is useful to separate
    this out from 3-42 for noise source analysis purposes and because sometimes
    dark current is defined in another way.

    :param jd:
        dark current density (A/m^2)
    :param w_x:
        the 1/e blur spot radii in the x direction
    :param w_y:
        the 1/e blur spot radii in the y direction

    :return:
        jde :
            dark current electron rate (e-/s); for TDI systems, just multiply
            the result by the number of TDI stages
    """
    jde = jd * w_x * w_y / qc  # recall that qc is defined as charge of an electron
    return jde


def image_domain_defocus_radii(D: float, dz: float, f: float) -> float:  # noqa: N803
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


def slice_otf(otf: np.ndarray, ang: float) -> np.ndarray:
    """Returns a one dimensional slice of a 2D OTF (or MTF) along the direction specified by the input angle.

    :param otf:
        OTF defined by spatial frequencies (u,v) (unitless)
    :param ang:
        slice angle (radians); a 0 radian slice is along the u axis.  The
        angle rotates counterclockwise. Angle pi/2 is along the v axis.
    :return:
        o_slice:
            one-dimensional OTF in the direction of angle; the sample spacing
            of o_slice is the same as the original otf
    """
    u = np.linspace(-1.0, 1.0, otf.shape[0])
    v = np.linspace(1.0, -1.0, otf.shape[1])
    r = np.arange(0.0, 1.0, u[1] - u[0])

    f = interpolate.interp2d(u, v, otf)
    o_slice = np.diag(f(r * np.cos(ang), r * np.sin(ang)))
    # the interpolator, f, calculates a bunch of points that we don't really
    # need since everything but the diagonal is thrown away.  It works but
    # it's inefficient.

    return o_slice


def apply_otf_to_image(
    ref_img: np.ndarray,
    ref_gsd: float,
    ref_range: float,
    otf: np.ndarray,
    df: float,
    ifov: float,
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
    psf = otf_to_psf(otf, df, 2 * np.arctan(ref_gsd / 2 / ref_range))

    # filter the image
    blur_img = cv2.filter2D(ref_img, -1, psf)

    # resample the image to the camera's ifov
    sim_img = resample_2D(blur_img, ref_gsd / ref_range, ifov)

    # resample psf (good for health checks on the simulation)
    sim_psf = resample_2D(psf, ref_gsd / ref_range, ifov)

    return sim_img, sim_psf


def common_OTFs(  # noqa: N802
    sensor: Sensor,
    scenario: Scenario,
    uu: np.ndarray,
    vv: np.ndarray,
    mtf_wavelengths: np.ndarray,
    mtf_weights: np.ndarray,
    slant_range: float,
    int_time: float,
) -> OTF:
    """Returns optical transfer functions for the most common sources.

    This code originally served the NIIRS model but has been abstracted for other
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
    :param mtf_wavelengths:
        a numpy array of wavelengths (m)
    :param mtf_weights:
        a numpy array of weights for each wavelength contribution (arb)
    :param slant_range:
        distance between the sensor and the target (m)
    :param int_time:
        integration time (s)

    :return:
        otf:
            an object containing results of the OTF calculations along with
            many intermediate calculations; the full system OTF is contained
            in otf.system_OTF

    :raises:
        ZeroDivisionError:
            if slant_range is 0
        IndexError:
            if uu or vv are empty
    """
    otf = OTF()

    # aperture OTF
    ap_function = lambda wavelengths: circular_aperture_OTF(  # noqa: E731
        uu, vv, wavelengths, sensor.D, sensor.eta
    )
    otf.ap_OTF = weighted_by_wavelength(mtf_wavelengths, mtf_weights, ap_function)

    # turbulence OTF
    if (
        scenario.cn2_at_1m > 0.0
    ):  # this option allows you to turn off turbulence completely
        # by setting cn2 at the ground level to 0
        otf.turb_OTF, otf.r0_band = polychromatic_turbulence_OTF(
            uu,
            vv,
            mtf_wavelengths,
            mtf_weights,
            scenario.altitude,
            slant_range,
            sensor.D,
            scenario.ha_wind_speed,
            scenario.cn2_at_1m,
            int_time * sensor.n_tdi,
            scenario.aircraft_speed,
        )
    else:
        otf.turb_OTF = np.ones(uu.shape)
        otf.r0_band = 1e6 * np.ones(uu.shape)

    # detector OTF
    otf.det_OTF = detector_OTF(uu, vv, sensor.w_x, sensor.w_y, sensor.f)

    # jitter OTF
    otf.jit_OTF = jitter_OTF(uu, vv, sensor.s_x, sensor.s_y)

    # drift OTF
    otf.drft_OTF = drift_OTF(
        uu,
        vv,
        sensor.da_x * int_time * sensor.n_tdi,
        sensor.da_y * int_time * sensor.n_tdi,
    )

    # wavefront OTF
    wav_function = lambda wavelengths: wavefront_OTF(  # noqa: E731
        uu,
        vv,
        wavelengths,
        sensor.pv * (sensor.pv_wavelength / wavelengths) ** 2,
        sensor.L_x,
        sensor.L_y,
    )
    otf.wav_OTF = weighted_by_wavelength(mtf_wavelengths, mtf_weights, wav_function)

    # filter OTF (e.g. a sharpening filter but it could be anything)
    if sensor.filter_kernel.shape[0] > 1:
        # note that we're assuming equal ifovs in the x and y directions
        otf.filter_OTF = filter_OTF(uu, vv, sensor.filter_kernel, sensor.p_x / sensor.f)
    else:
        otf.filter_OTF = np.ones(uu.shape)

    # system OTF
    otf.system_OTF = (
        otf.ap_OTF
        * otf.turb_OTF
        * otf.det_OTF
        * otf.jit_OTF
        * otf.drft_OTF
        * otf.wav_OTF
        * otf.filter_OTF
    )

    return otf


def resample_2D(  # noqa: N802
    img_in: np.ndarray, dx_in: float, dx_out: float
) -> np.ndarray:
    """Resample an image.

    :param img:
        the input image
    :param dx_in:
        sample spacing of the input image (radians)
    :param dx_out:
        sample spacing of the output image (radians)

    :return:
        img_out:
            output image

    :raises:
        IndexError:
            if imigin is not a 2D array
        ZeroDivisionError:
            if dx_out is 0
        cv2.error:
            if dx_in is 0

    """
    new_x = int(np.round(img_in.shape[1] * dx_in / dx_out))
    new_y = int(np.round(img_in.shape[0] * dx_in / dx_out))
    img_out = cv2.resize(img_in, (new_x, new_y))

    return img_out
