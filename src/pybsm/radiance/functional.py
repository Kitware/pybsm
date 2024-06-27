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

This module deals with all things related to radiance, including black body
spectral emission calculations, spectral atmospheric absorption, and conversion
of radiance to photons. Anything material with a non-zero temperature emits
electromagnetic radiation according to Planck's law and its temperature,
spectral emissivity.
"""
# standard library imports
import inspect
import logging
import os
import warnings
from typing import Tuple

# 3rd party imports
import numpy as np

# local imports
from pybsm import noise
from pybsm.simulation.sensor import Sensor

from .snr_metrics import SNRMetrics

# new in version 0.2.  We filter warnings associated with calculations in the
# function circularApertureOTF.  These invalid values are caught as NaNs and
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


def at_focal_plane_irradiance(
    D: float, f: float, L: np.ndarray  # noqa: N803
) -> np.ndarray:
    """Converts pupil plane radiance to focal plane irradiance for an extended source.

    This is a variation on part of IBSM Equation 3-34.  There is one
    modification: the IBSM conversion factor pi/(4(f/#)^2) is replaced with
    pi/(1+ 4(f/#)^2), which is valid over a wider range of f-numbers (source:
    John Schott,"Remote Sensing: The Image Chain Approach," Oxford University
    Press, 1997). If the telescope is obscured, E is further reduced by
    1-eta**2, where eta is the relative linear obscuration.

    :param D:
        effective aperture diameter (m)
    :type D: float
    :param f:
        focal length (m)
    :type f: float
    :param L:
        total radiance (W/m^2 sr) or spectral radiance (W/m^2 sr m)
    :type L: float

    :return:
        E:
            total irradiance (W/m^2) or spectral irradiance (W/m^2 m) at the focal plane

    :raises:
        ZeroDivisionError:
            if D is 0

    """
    if L.size == 0:
        warnings.warn(
            UserWarning("Input array is empty. Expect output to be empty"), stacklevel=2
        )

    E = L * np.pi / (1.0 + 4.0 * (f / D) ** 2.0)  # noqa: N806

    return E


def blackbody_radiance(lambda0: np.ndarray, T: float) -> np.ndarray:  # noqa: N803
    """Calculates blackbody spectral radiance.  IBSM Equation 3-35.

    :param lambda0:
        wavelength (m)
    :type lambda0: float
    :param T:
        temperature (K)
    :type T: float

    :return:
        Lbb :
            blackbody spectral radiance (W/m^2 sr m)
    """
    # lambda0 = lambda0+1e-20
    if lambda0.size == 0:
        warnings.warn(
            UserWarning("Input array is empty. Expect output to be empty"), stacklevel=2
        )

    Lbb = (2.0 * hc * cc**2.0 / lambda0**5.0) * (  # noqa: N806
        np.exp(hc * cc / (lambda0 * kc * T)) - 1.0
    ) ** (-1.0)

    return Lbb


def check_well_fill(total_photoelectrons: float, max_fill: float) -> float:
    """Check to see if the total collected photoelectrons are greater than the desired maximum well fill.

    If so, provide a scale factor to reduce the integration time.

    :param total_photoelectrons:
        array of wavelengths (m)
    :type totalPhotelectrons: np.array
    :param max_fill:
        desired well fill; i.e., maximum well size x desired fill fraction
    :type max_fill: float

    :return:
        scale_factor:
            the new integration time is scaled by scale_factor
    """
    scale_factor = 1.0
    if total_photoelectrons > max_fill:
        scale_factor = max_fill / total_photoelectrons
    return scale_factor


def cold_shield_self_emission(
    wavelengths: np.ndarray,
    cold_shield_temperature: float,
    D: float,  # noqa: N803
    f: float,
) -> np.ndarray:
    """Represents spectral irradiance on the FPA due to emissions from the walls of the dewar itself.

    For infrared systems, this term represents spectral irradiance on the FPA due to emissions from
    the walls of the dewar itself.

    :param wavelengths:
        wavelength array (m)
    :param cold_shield_temperature:
        temperature of the cold shield (K); it is a common approximation to
        assume that the coldshield is at the same temperature as the detector
        array
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        cold_shield_E:
            cold shield spectral irradiance at the FPA (W / m^2 m)

    :raises:
        ZeroDivisionError:
            if D is 0
    """
    if wavelengths.size == 0:
        warnings.warn(
            UserWarning("Input array is empty. Expect output to be empty"), stacklevel=2
        )
    # coldshield solid angle x blackbody emitted radiance
    cold_shield_E = (  # noqa: N806
        np.pi - np.pi / (4.0 * (f / D) ** 2.0 + 1.0)
    ) * blackbody_radiance(wavelengths, cold_shield_temperature)

    return cold_shield_E  # noqa: N806


def cold_stop_self_emission(
    wavelengths: np.ndarray,
    cold_filter_temperature: float,
    cold_filter_emissivity: float,
    D: float,  # noqa: N803
    f: float,
) -> np.ndarray:
    """For infrared systems, this term represents spectral irradiance emitted by the cold stop on to the FPA.

    :param wavelengths:
        wavelength array (m)
    :param cold_filter_temperature:
        temperature of the cold filter; it is a common approximation to assume
        that the filter is at the same temperature as the detector array
    :param cold_filter_emissivity:
        emissivity through the cold filter (unitless); a common approximation
        is 1-cold filter transmission
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        cold_stop_E:
            optics emitted irradiance on to the FPA (W / m^2 m)

    :raises:
        ZeroDivisionError:
            if D is 0
    """
    if wavelengths.size == 0:
        warnings.warn(
            UserWarning("Input array is empty. Expect output to be empty"), stacklevel=2
        )

    cold_stop_L = cold_filter_emissivity * blackbody_radiance(  # noqa: N806
        wavelengths, cold_filter_temperature
    )
    cold_stop_E = at_focal_plane_irradiance(D, f, cold_stop_L)  # noqa: N806
    return cold_stop_E


def focal_plane_integrated_irradiance(
    L: np.ndarray,  # noqa: N803
    L_s: float,  # noqa: N803
    t_opt: float,
    e_opt: float,
    lambda0: np.ndarray,
    d_lambda: float,
    optics_temperature: float,
    D: float,  # noqa: N803
    f: float,
) -> np.ndarray:
    """IBSM Equation 3-34.

    Calculates band integrated irradiance at the focal plane, including at-aperture scene radiance, optical
    self-emission, and non-thermal stray radiance.  NOTE: this function is only included for completeness.
    It is much better to use spectral quantities throughout the modeling process.

    :param L:
        band integrated at-aperture radiance (W/m^2 sr)
    :param L_s:
        band integrated stray radiance from sources other than self emission
        (W/m^2 sr)
    :param t_opt:
        full system in-band optical transmission (unitless); if the telescope
        is obscured, t_opt is further reduced by 1-eta**2, where
        eta is the relative linear obscuration
    :param e_opt:
        full system in-band optical emissivity (unitless); 1-t_opt is a good
        approximation
    :param lambda0:
        wavelength at the center of the system bandpass (m)
    :param d_lambda:
        system spectral bandwidth (m)
    :param optics_temperature:
        temperature of the optics (K)
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        E:
            integrated irradiance (W/m^2) at the focal plane
    """
    L = (  # noqa: N806
        t_opt * L
        + e_opt * blackbody_radiance(lambda0, optics_temperature) * d_lambda
        + L_s
    )  # noqa: N806
    E = at_focal_plane_irradiance(D, f, L)  # noqa: N806

    return E


def optics_self_emission(
    wavelengths: np.ndarray,
    optics_temperature: float,
    optics_emissivity: float,
    cold_filter_transmission: float,
    D: float,  # noqa: N803
    f: float,
) -> np.ndarray:
    """This term represents spectral irradiance emitted by the optics (but not the cold stop) on to the FPA.

    For infrared systems, this term represents spectral irradiance emitted by the
    optics (but not the cold stop) on to the FPA.

    :param wavelengths:
        wavelength array (m)
    :param optics_temperature:
        temperature of the optics (K)
    :param optics_emissivity:
        emissivity of the optics (unitless) except for the cold filter;
        a common approximation is 1-optics transmissivity
    :param cold_filter_transmission:
        transmission through the cold filter (unitless)
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        optics_E:
            optics emitted irradiance on to the FPA (W / m^2 m)

    :raises:
        ZeroDivisionError:
            if D is 0
    """
    if wavelengths.size == 0:
        warnings.warn(
            UserWarning("Input array is empty. Expect output to be empty"), stacklevel=2
        )

    optics_L = (  # noqa: N806
        cold_filter_transmission
        * optics_emissivity
        * blackbody_radiance(wavelengths, optics_temperature)
    )
    optics_E = at_focal_plane_irradiance(D, f, optics_L)  # noqa: N806
    return optics_E


def photon_detection_rate(
    E: np.ndarray,  # noqa: N803
    w_x: float,
    w_y: float,
    wavelengths: np.ndarray,
    qe: np.ndarray,  # noqa: N806
) -> np.ndarray:
    """IBSM Equation 3-42 with dark current, integration time, and tdi separated out.

    Conversion of photons into photoelectrons.  There is a possible disconnect here in the documentation.
    Equation 3-42 appears to be a spectral quantity but the documentation calls for an integrated
    irradiance.  It is definitely used here as a spectral quantity.

    :param E:
        spectral irradiance (W/m^2 m) at the focal plane at each wavelength
    :param w_x:
        detector size (width) in the x direction (m)
    :param w_y:
        detector size (width) in the y direction (m)
    :param wavelengths:
        wavelength array (m)
    :param qe:
        quantum efficiency (e-/photon)

    :return:
        dN:
            array of photoelectrons/wavelength/second (e-/m)
    :NOTE:
        To calculate total integrated photoelectrons, N = td*n_tdi*np.trapz(dN,
        wavelens) where td is integration time (s) and n_tdi is the number of
        tdi stages (optional).
    """
    if E.size == 0 or wavelengths.size == 0 or qe.size == 0:
        warnings.warn(
            UserWarning("Input array(s) are empty. Expect output to be empty"),
            stacklevel=2,
        )

    dN = (wavelengths / (hc * cc)) * qe * w_x * w_y * E  # noqa: N806
    return dN


def photon_detector_SNR(  # noqa: N802
    sensor: Sensor,
    radiance_wavelengths: np.ndarray,
    target_radiance: np.ndarray,
    background_radiance: np.ndarray,
) -> SNRMetrics:
    """Calculates extended target contrast SNR for semiconductor-based photon detector systems.

    Calculates extended target contrast SNR for semiconductor-based photon detector
    systems (as opposed to thermal detectors).  This code originally
    served the NIIRS model but has been abstracted for other uses.  Photon,
    dark current, quantization, and read noise are all explicitly considered.
    You can also pass in other noise terms (as rms photoelectrons) as a numpy
    array sensor.other_noise.

    :param sensor:
        an object from the class sensor
    :param radiance_wavelengths:
        a numpy array of wavelengths (m)
    :param target_radiance:
        a numpy array of target radiance values corresponding to
        radiance_wavelengths (W/m^2 sr m)
    :param background_radiance:
        a numpy array of target radiance values corresponding to
        radiance_wavelengths (W/m^2 sr m)

    :return:
        snr:
            an object containing results of the SNR calculation along with many
            intermediate calculations; the SNR value is contained in snr.snr
    """
    snr = SNRMetrics("signal-to-noise calculation")

    # resample the optical transmission and quantum efficiency functions
    snr.opt_trans = (
        sensor.cold_filter_transmission
        * (1.0 - sensor.eta**2)
        * resample_by_wavelength(
            sensor.opt_trans_wavelengths,
            sensor.optics_transmission,
            radiance_wavelengths,
        )
    )
    snr.qe = resample_by_wavelength(
        sensor.qe_wavelengths, sensor.qe, radiance_wavelengths
    )

    # for infrared systems, calculate FPA irradiance contributions from within
    # the sensor system itself
    snr.other_irradiance = (
        cold_shield_self_emission(
            radiance_wavelengths,
            sensor.cold_shield_temperature,
            sensor.D,
            sensor.f,
        )
        + optics_self_emission(
            radiance_wavelengths,
            sensor.optics_temperature,
            sensor.optics_emissivity,
            sensor.cold_filter_transmission,
            sensor.D,
            sensor.f,
        )
        + cold_stop_self_emission(
            radiance_wavelengths,
            sensor.cold_filter_temperature,
            sensor.cold_filter_emissivity,
            sensor.D,
            sensor.f,
        )
    )

    # initial estimate of  total detected target and background photoelectrons
    # first target.  Note that snr.weights is useful for later calculations
    # that require weighting as a function of wavelength (e.g. aperture OTF)
    snr.tgt_n_rate, snr.tgt_FPA_irradiance, snr.weights = signal_rate(
        radiance_wavelengths,
        target_radiance,
        snr.opt_trans,
        sensor.D,
        sensor.f,
        sensor.w_x,
        sensor.w_y,
        snr.qe,
        snr.other_irradiance,
        sensor.dark_current,
    )
    snr.tgt_n = snr.tgt_n_rate * sensor.int_time * sensor.n_tdi
    # then background
    snr.bkg_n_rate, snr.bkg_FPA_irradiance, _ = signal_rate(
        radiance_wavelengths,
        background_radiance,
        snr.opt_trans,
        sensor.D,
        sensor.f,
        sensor.w_x,
        sensor.w_y,
        snr.qe,
        snr.other_irradiance,
        sensor.dark_current,
    )
    snr.bkg_n = snr.bkg_n_rate * sensor.int_time * sensor.n_tdi

    # check to see that well fill is within a desirable range and, if not,
    # scale back the integration time and recalculate the total photon counts
    scale_factor = check_well_fill(
        np.max([snr.tgt_n, snr.bkg_n]), sensor.max_well_fill * sensor.max_n
    )
    snr.tgt_n = scale_factor * snr.tgt_n
    snr.bkg_n = scale_factor * snr.bkg_n
    snr.int_time = scale_factor * sensor.int_time
    snr.well_fraction = np.max([snr.tgt_n, snr.bkg_n]) / sensor.max_n
    # another option would be to reduce TDI stages if applicable, this should
    # be a concern if TDI mismatch MTF is an issue

    # calculate contrast signal (i.e. target difference above or below the
    # background)
    snr.contrast_signal = snr.tgt_n - snr.bkg_n

    # break out noise terms (rms photoelectrons)
    # signal_noise includes scene photon noise, dark current noise, and self
    # emission noise
    snr.signal_noise = np.sqrt(np.max([snr.tgt_n, snr.bkg_n]))
    # just noise from dark current
    snr.dark_current_noise = np.sqrt(sensor.n_tdi * sensor.dark_current * snr.int_time)
    # quantization noise
    snr.quantization_noise = noise.quantization_noise(sensor.max_n, sensor.bit_depth)
    # photon noise due to self emission in the optical system
    snr.self_emission_noise = np.sqrt(
        np.trapz(
            photon_detection_rate(
                snr.other_irradiance,
                sensor.w_x,
                sensor.w_y,
                radiance_wavelengths,
                snr.qe,
            ),
            radiance_wavelengths,
        )
        * snr.int_time
        * sensor.n_tdi
    )

    # note that signal_noise includes sceneNoise, dark current noise, and self
    # emission noise
    snr.total_noise = np.sqrt(
        snr.signal_noise**2
        + snr.quantization_noise**2
        + sensor.read_noise**2
        + np.sum(sensor.other_noise**2)
    )

    # calculate signal-to-noise ratio
    snr.snr = snr.contrast_signal / snr.total_noise

    return snr


def reflectance_to_photoelectrons(
    atm: np.ndarray, sensor: Sensor, int_time: float, target_temp: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provides a mapping between reflectance on the ground and photoelectrons collected in the sensor well.

    Provides a mapping between reflectance (0 to 1 in 100 steps) on the
    ground and photoelectrons collected in the sensor well for a given
    atmosphere. The target is assumed to be fully illuminated and spectrally
    flat. The target reflects the solar illumination at a rate corresponding to
    its assumed reflectance. But, this function also considers the finite
    temperature of the target and the radiative emission from the target
    itself. This function also considers the thermal emission from the imaging
    system itself (cold shield, cold stop, optical glass), which is relevant in
    system that are sensitive to the thermal emission wavebands. Dark current
    is included.

    :param atm:
        atmospheric data as defined in utils.loadDatabaseAtmosphere; the slant
        range between the target and sensor are implied by this choice
    :param sensor:
        sensor parameters as defined in the pybsm sensor class
    :param int_time:
        camera integration time (s)
    :param target_temp: float
        Temperature of the target (Kelvin)

    :return:
        ref :
            array of reflectance values (unitless) from 0 to 1 in 100 steps
        pe :
            photoelectrons generated during the integration time corresponding
            to the reflectance values in ref
        spectral_weights :
            2xN arraylike details of the relative spectral contributions to the collected
            signal, which is useful for wavelength-weighted OTF calculations;
            the first row is the wavelength (m) and the second row is the
            relative contribution to the signal from the associated
            column-paired wavelength

    :raises:
        IndexError:
            if atm is not a 2D array
    """
    ref = np.linspace(0.0, 1.0, 100)
    pe = np.zeros(ref.shape)
    atm = atm[atm[:, 0] >= sensor.opt_trans_wavelengths[0], :]
    atm = atm[atm[:, 0] <= sensor.opt_trans_wavelengths[-1], :]

    for idx in np.arange(ref.size):
        # Calculate the total radiance from the target including both
        # reflection of the solar illumination and the radiative emission from
        # the object assumed to be at 300 K.
        target_radiance = total_radiance(atm, ref[idx], 300.0)

        wavelengths = atm[:, 0]

        opt_trans = (
            sensor.cold_filter_transmission
            * (1.0 - sensor.eta**2)
            * resample_by_wavelength(
                sensor.opt_trans_wavelengths,
                sensor.optics_transmission,
                wavelengths,
            )
        )

        qe = resample_by_wavelength(sensor.qe_wavelengths, sensor.qe, wavelengths)

        # The components of the imaging system is at a non-zero temperature and
        # itself generates radiative emissions. So, we account for these
        # emissions here. This is only relevant in the thermal infrared bands.
        other_irradiance = cold_shield_self_emission(
            wavelengths, sensor.cold_shield_temperature, sensor.D, sensor.f
        )
        other_irradiance = other_irradiance + optics_self_emission(
            wavelengths,
            sensor.optics_temperature,
            sensor.optics_emissivity,
            sensor.cold_filter_transmission,
            sensor.D,
            sensor.f,
        )
        other_irradiance = other_irradiance + cold_stop_self_emission(
            wavelengths,
            sensor.cold_filter_temperature,
            sensor.cold_filter_emissivity,
            sensor.D,
            sensor.f,
        )

        tgt_n_rate, tgt_FPA_irradiance, weights = signal_rate(  # noqa: N806
            wavelengths,
            target_radiance,
            opt_trans,
            sensor.D,
            sensor.f,
            sensor.w_x,
            sensor.w_y,
            qe,
            other_irradiance,
            sensor.dark_current,
        )

        pe[idx] = tgt_n_rate * int_time * sensor.n_tdi

    sat = pe.max() / sensor.max_n
    if sat > 1:
        logging.info(
            f"Reducing integration time from {int_time} to {int_time/sat}"
            " to avoid overexposure"
        )
        pe = pe / sat

    # Clip to the maximum number of photoelectrons that can be held.
    pe[pe > sensor.max_n] = sensor.max_n

    spectral_weights = np.vstack([wavelengths, weights / max(weights)])

    return ref, pe, spectral_weights


def signal_rate(
    wavelengths: np.ndarray,
    target_radiance: np.ndarray,
    optical_transmission: np.ndarray,
    D: float,  # noqa: N803
    f: float,
    w_x: float,
    w_y: float,
    qe: np.ndarray,
    other_irradiance: np.ndarray,
    dark_current: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """For semiconductor-based detectors, returns the signal rate generated at the output as well as other quantities.

    For semiconductor-based detectors, returns the signal rate (total
    photoelectrons/s) generated at the output of the detector along with a
    number of other related quantities.  Multiply this quantity by the
    integration time (and the number of TDI stages, if applicable) to determine
    the total number of detected photoelectrons.

    :param wavelengths:
        array of wavelengths (m)
    :param target_radiance:
        apparent target spectral radiance at the aperture including all
        atmospheric contributions (W/sr m^2 m)
    :param background_radiance:
        apparent background spectral radiance at the aperture including all
        atmospheric contributions (W/sr m^2 m)
    :param optical_transmission:
        transmission of the telescope optics as a function of wavelength
        (unitless)
    :param D:
        effective aperture diameter (m)
    :param (w_x,w_y):
        detector size (width) in the x and y directions (m)
    :param f:
        focal length (m)
    :param qe:
        quantum efficiency as a function of wavelength (e-/photon)
    :param other_irradiance:
        spectral irradiance from other sources (W/m^2 m);
        particularly useful for self emission in infrared cameras
        and may also represent stray light
    :param dark_current:
        detector dark current (e-/s)

    :return:
        tgt_rate:
            total integrated photoelectrons per seconds (e-/s)
        tgt_FPA_irradiance:
            spectral irradiance at the FPA (W/m^2 m)
        tgt_dN:
            spectral photoelectrons (e-/s m)

    :raises:
        ZeroDivisionError:
            if D is 0

    """
    if (
        wavelengths.size == 0
        or target_radiance.size == 0
        or optical_transmission.size == 0
        or other_irradiance.size == 0
    ):
        warnings.warn(
            UserWarning("Input array(s) are empty. Expect output to be empty"),
            stacklevel=2,
        )

    # get at FPA spectral irradiance
    tgt_FPA_irradiance = (  # noqa: N806
        optical_transmission * at_focal_plane_irradiance(D, f, target_radiance)
        + other_irradiance
    )

    # convert spectral irradiance to spectral photoelectron rate
    tgt_dN = photon_detection_rate(  # noqa: N806
        tgt_FPA_irradiance, w_x, w_y, wavelengths, qe
    )

    # calculate total detected target and background photoelectron rate
    tgt_rate = np.trapz(tgt_dN, wavelengths) + dark_current

    return tgt_rate, tgt_FPA_irradiance, tgt_dN


def total_radiance(
    atm: np.ndarray, reflectance: float, temperature: float
) -> np.ndarray:
    """Calculates total spectral radiance at the aperture for a object of interest.

    :param atm:
        matrix of atmospheric data (see utils.loadDatabaseAtmosphere for details)
    :param reflectance:
        object reflectance (unitless)
    :param temperature:
        object temperature (Kelvin)

    :return:
        radiance:
            radiance = path thermal + surface emission + solar scattering +
            ground reflected (W/m^2 sr m)

    :raises:
        IndexError:
            if atm is not a 2D array

    :NOTE:
        In the emissive infrared region (e.g. >= 3 micrometers), the nighttime
        case is very well approximated by subtracting off atm[:,4] from the
        total spectral radiance.
    """
    db_reflectance = 0.15  # object reflectance used in the database
    radiance = (
        atm[:, 2]
        + (1.0 - reflectance) * blackbody_radiance(atm[:, 0], temperature) * atm[:, 1]
        + atm[:, 4]
        + atm[:, 5] * (reflectance / db_reflectance)
    )

    return radiance


def giqe_radiance(atm: np.ndarray, is_emissive: int) -> Tuple[np.ndarray, np.ndarray]:
    """This function provides target and background spectral radiance as defined by the GIQE.

    :param atm:
        an array containing the following data:
        atm[:,0] - wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps

        atm[:,1] - (TRANS) total transmission through the defined path

        atm[:,2] - (PTH THRML) radiance component due to atmospheric emission
        and scattering received at the observer

        atm[:,3] - (SURF EMIS) component of radiance due to surface emission
        received at the observer

        atm[:,4] - (SOL SCAT) component of scattered solar radiance received
        at the observer

        atm[:,5] - (GRND RFLT) is the total solar flux impingement on the
        ground and reflected directly to the sensor from the ground. (direct
        radiance + diffuse radiance) * surface reflectance

        NOTE: Units for columns 1 through 5 are in radiance W/(sr m^2 m).
    :param is_emissive:
        is_emissive = 1 for thermal emissive band NIIRS, otherwise
        is_emissive = 0

    :return:
        target_radiance:
            apparent target spectral radiance at the aperture including all
            atmospheric contributions
        background_radiance:
            apparent background spectral radiance at the aperture including
            all atmospheric contributions

    :NOTE:
        The nighttime emissive case is well approximated by subtracting off
        atm[:,4] from the returned values.

    """
    tgt_temp = 282.0  # target temperature (original GIQE suggestion was 282 K)
    bkg_temp = 280.0  # background temperature (original GIQE 280 K)
    tgt_ref = 0.15  # percent reflectance of the target (should be .15 for GIQE)
    bkg_ref = 0.07  # percent reflectance of the background (should be .07 for GIQE)

    if is_emissive:
        # target and background are blackbodies
        target_radiance = total_radiance(atm, 0.0, tgt_temp)
        background_radiance = total_radiance(atm, 0.0, bkg_temp)
    else:
        target_radiance = total_radiance(atm, tgt_ref, tgt_temp)
        background_radiance = total_radiance(atm, bkg_ref, bkg_temp)

    return target_radiance, background_radiance


def resample_by_wavelength(
    wavelengths: np.ndarray, values: np.ndarray, new_wavelengths: np.ndarray
) -> np.ndarray:
    """Resamples arrays that are input as a function of wavelength.

    :param wavelengths:
        array of wavelengths (m)
    :param values:
        array of values to be resampled (arb)
    :param new_wavelengths:
        the desired wavelength range and step size (m)

    :return:
        new_values
            array of values resampled to match new_wavelengths;
            extrapolated values are set to 0

    :raises:
        ValueError:
            if the length of wavelengths array and values array are
            not equal
    """
    new_values = np.interp(new_wavelengths, wavelengths, values, 0.0, 0.0)
    return new_values
