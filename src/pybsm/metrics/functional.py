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

# standard library imports
import inspect
import os
import warnings
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

# local imports
import pybsm.otf as otf
import pybsm.radiance as radiance
from pybsm import geospatial, noise, utils
from pybsm.simulation import Scenario, Sensor

from .metrics import Metrics

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


def giqe3(rer: float, gsd: float, eho: float, ng: float, snr: float) -> float:
    """IBSM Equation 3-56.  The General Image Quality Equation version 3.0.

        The GIQE returns values on the National Image Interpretability Rating
        Scale. Note: geometric mean values are simply sqrt(value_x * value_y),
        where x and y are orthogonal directions in the image.

    :param rer:
        geometric mean relative edge response (unitless)
    :param gsd:
        geometric mean ground sample distance (m)
    :param eho:
        geometric mean edge height overshoot (unitless)
    :param ng:
        noises gain, i.e. increase in noise due to sharpening (unitless); if no
        sharpening is applied, then ng = 1
    :param snr:
        contrast signal-to-noise ratio (unitless)

    :return:
        niirs :
            a National Image Interpretability Rating Scale value (unitless)
    """
    niirs = 11.81 + 3.32 * np.log10(rer / (gsd / 0.0254)) - 1.48 * eho - ng / snr
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion
    return niirs


def giqe4(
    rer: float, gsd: float, eho: float, ng: float, snr: float, elev_angle: float
) -> Tuple[float, float]:
    """General Image Quality Equation version 4 from Leachtenauer, et al.

    "General Image Quality Equation: GIQE," Applied Optics, Vol 36, No 32,
    1997. The use of GIQE 4 is not endorsed but it is added to pyBSM for
    historical completeness.

    :param rer:
        geometric mean relative edge response (unitless)
    :param gsd:
        geometric mean ground sample distance (m) before projection into the
        ground plane
    :param eho:
        geometric mean edge height overshoot (unitless)
    :param ng:
        noises gain, i.e. increase in noise due to sharpening (unitless); if no
        sharpening is applied then ng = 1
    :param snr:
        contrast signal-to-noise ratio (unitless)
    :param elev_angle:
        sensor elevation angle as measured from the target (rad), i.e.
        pi/2-nadir_angle. See pybsm.geospatial.nadir_angle for more information. Note that
        the GIQE4 paper defines this angle differently but we stick with this
        version to be consistent with the GIQE 5 code. The outcome is the same
        either way.

    :return:
        niirs :
            a National Image Interpretability Rating Scale value (unitless)
    """
    if rer >= 0.9:
        c_1 = 3.32
        c_2 = 1.559
    else:
        c_1 = 3.16
        c_2 = 2.817

    gsd_gp = gsd / (np.sin(elev_angle) ** (0.5))  # note that the exponent captures the
    # fact that only one direction in the gsd is distorted by projection into
    # the ground plane

    niirs = (
        10.251
        - c_1 * np.log10(gsd_gp / 0.0254)
        + c_2 * np.log10(rer)
        - 0.656 * eho
        - 0.334 * ng / snr
    )
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion
    return niirs, gsd_gp


def giqe5(
    rer_1: float, rer_2: float, gsd: float, snr: float, elev_angle: float
) -> Tuple[float, float, float]:
    """NGA The General Image Quality Equation version 5.0. 16 Sep 2015.

    https://gwg.nga.mil/ntb/baseline/docs/GIQE-5_for_Public_Release.pdf
    This version of the GIQE replaces the earlier versions and should be used
    in all future analyses.  See also "Airborne Validation of the General Image
    Quality Equation 5"
    https://www.osapublishing.org/ao/abstract.cfm?uri=ao-59-32-9978

    :param rer_1:
        relative edge response in two directions (e.g., along- and across-
        scan, horizontal and vertical, etc.); see also pybsm.metrics.functional.giqe5_RER.
        (unitless)
    :param rer_2:
        relative edge response in two directions (e.g., along- and across-
        scan, horizontal and vertical, etc.); see also pybsm.metrics.functional.giqe5_RER.
        (unitless)
    :param gsd:
        image plane geometric mean ground sample distance (m), as defined for
        GIQE3; the GIQE 5 version of GSD is calculated within this function
    :param snr:
        contrast signal-to-noise ratio (unitless), as defined for GIQE3
    :param elev_angle:
        sensor elevation angle as measured from the target (rad), i.e.
        pi/2-nadir_angle; see pybsm.geospatial.nadir_angle for more information

    :return:
        niirs :
            a National Image Interpretability Rating Scale value (unitless)
        gsd_w :
            elevation angle weighted GSD (m)
        rer :
            weighted relative edge response (rer)
    :NOTE:
        NIIRS 5 test case: rer_1=rer_2=0.35, gsd = 0.52832 (20.8 inches),
        snr = 50, elev_angle = np.pi/2. From Figure 1 in the NGA GIQE5 paper.

    """
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion in the niirs equation below
    gsd_w = gsd / (
        np.sin(elev_angle) ** (0.25)
    )  # geometric mean of the image plane and ground plane gsds

    rer = (np.max([rer_1, rer_2]) * np.min([rer_1, rer_2]) ** 2.0) ** (1.0 / 3.0)

    niirs = (
        9.57
        - 3.32 * np.log10(gsd_w / 0.0254)
        + 3.32 * (1 - np.exp(-1.9 / snr)) * np.log10(rer)
        - 2.0 * np.log10(rer) ** 4.0
        - 1.8 / snr
    )
    return niirs, gsd_w, rer


def giqe5_RER(  # noqa: N802
    mtf: np.ndarray, df: float, ifov_x: float, ifov_y: float
) -> Tuple[float, float]:
    """Calculates the relative edge response from a 2-D MTF.

    This function is primarily for use with the GIQE 5. It implements IBSM equations 3-57 and
    3-58.  See pybsm.metrics.functional.giqe_edge_terms for the GIQE 3 version.

    :param mtf:
        2-dimensional full system modulation transfer function (unitless);
        MTF is the magnitude of the OTF
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifov_x:
        x-direction instantaneous field-of-view of a detector (radians)
    :param ifov_y:
        y-direction instantaneous field-of-view of a detector (radians)

    :return:
        rer_0:
            relative edge response at 0 degrees orientation (unitless)
        rer_90:
            relative edge response at 90 degrees orientation (unitless)
    """
    u_slice = otf.functional.slice_otf(mtf, 0)
    v_slice = otf.functional.slice_otf(mtf, np.pi / 2)

    rer_0 = relative_edge_response(u_slice, df, ifov_x)
    rer_90 = relative_edge_response(v_slice, df, ifov_y)

    return rer_0, rer_90


def giqe_edge_terms(
    mtf: np.ndarray, df: float, ifov_x: float, ifov_y: float
) -> Tuple[float, float]:
    """Calculates the geometric mean relative edge response and edge high overshoot,from a 2-D MTF.

    This function is primarily for use with the GIQE. It implements IBSM equations 3-57 and 3-58.

    :param mtf:
        2-dimensional full system modulation transfer function (unitless);
        MTF is the magnitude of the OTF
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifov_x:
        x-direction instantaneous field-of-view of a detector (radians)
    :param ifov_y:
        y-direction instantaneous field-of-view of a detector (radians)

    :return:
        rer:
            geometric mean relative edge response (unitless)
        eho:
            geometric mean edge height overshoot (unitless)
    """
    u_slice = otf.functional.slice_otf(mtf, 0)
    v_slice = otf.functional.slice_otf(mtf, np.pi / 2)

    mtf_slice_er = relative_edge_response(u_slice, df, ifov_x)
    v_rer = relative_edge_response(v_slice, df, ifov_y)
    rer = np.sqrt(mtf_slice_er * v_rer)

    u_eho = edge_height_overshoot(u_slice, df, ifov_x)
    v_eho = edge_height_overshoot(v_slice, df, ifov_y)
    eho = np.sqrt(u_eho * v_eho)

    return rer, eho


def ground_resolved_distance(
    mtf_slice: np.ndarray, df: float, snr: float, ifov: float, slant_range: float
) -> float:
    """IBSM Equation 3-54.

    The ground resolved distance is the period of the smallest square wave pattern that can be resolved in an image.
    GRD can be limited by the detector itself (in which case GRD = 2*GSD) but, in general
    is a function of the system MTF and signal-to-noise ratio.

    :param mtf_slice:
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0
        cycles/radian
    :param df:
        spatial frequency step size (cycles/radian)
    :param snr:
        contrast signal-to-noise ratio (unitless)
    :param ifov:
        instantaneous field-of-view of a detector (radians)
    :param slant_range:
        distance between the target and sensor (m)
    :return:
        grd:
            ground resolved distance (m)
    """
    w = df * np.arange(1.0 * mtf_slice.size)
    u_r = (
        np.interp(3.0 / snr, mtf_slice[::-1], w[::-1])
        + 1e-12  # 1e-12 prevents division by zero in grd_cases
    )  # arrays were reversed to satisfy the requirements of np.interp

    grd_cases = slant_range * np.array([1.0 / u_r, 2.0 * ifov])
    grd = np.max(grd_cases)

    return grd


def niirs(sensor: Sensor, scenario: Scenario, interp: Optional[bool] = False) -> Metrics:
    """Returns NIIRS values and all intermediate calculations.

    This function implements the original MATLAB-based NIIRS model and can serve as a
    template for building other sensor models.

    :param sensor:
        an object from the class sensor
    :param scenario:
        an object from the class scenario

    :return:
        nm:
            an object containing results of the GIQE calculation along with
            many intermediate calculations
    """
    # initialize the output
    nm = Metrics("niirs " + sensor.name + " " + scenario.name)
    nm.sensor = sensor
    nm.scenario = scenario
    nm.slant_range = geospatial.curved_earth_slant_range(
        0.0, scenario.altitude, scenario.ground_range
    )

    # #########CONTRAST SNR CALCULATION#########
    # load the atmosphere model
    if interp:
        nm.atm = utils.load_database_atmosphere(
            scenario.altitude, scenario.ground_range, scenario.ihaze
        )
    else:
        nm.atm = utils.load_database_atmosphere_no_interp(
            scenario.altitude, scenario.ground_range, scenario.ihaze
        )
    # crop out out-of-band data (saves time integrating later)
    nm.atm = nm.atm[nm.atm[:, 0] >= nm.sensor.opt_trans_wavelengths[0], :]
    nm.atm = nm.atm[nm.atm[:, 0] <= nm.sensor.opt_trans_wavelengths[-1], :]

    if nm.sensor.opt_trans_wavelengths[0] >= 2.9e-6:
        is_emissive = 1  # toggle the GIQE to assume infrared imaging
        # the next four lines are bookkeeping for interpreting the results
        # since the giqe_radiance function assumes these values anyway
        nm.scenario.target_reflectance = 0.0
        nm.scenario.background_reflectance = 0.0
        nm.scenario.target_temperature = 282.0
        nm.scenario.target_temperature = 280.0

    else:
        is_emissive = 0
        # more bookkeeping (see previous comment)
        nm.scenario.target_reflectance = 0.15
        nm.scenario.background_reflectance = 0.07

    # get aperture radiances for reflective target and background with GIQE
    # type target parameters
    nm.tgt_radiance, nm.bkg_radiance = radiance.giqe_radiance(nm.atm, is_emissive)
    nm.radiance_wavelengths = nm.atm[:, 0]

    # now calculate now characteristics ****for a single frame******
    nm.snr = radiance.photon_detector_SNR(
        sensor, nm.radiance_wavelengths, nm.tgt_radiance, nm.bkg_radiance
    )

    # break out photon noise sources (not required for NIIRS but useful for
    # analysis) photon noise due to the scene itself (target,background, and
    # path emissions/scattering)
    tgt_noise = np.sqrt(
        np.trapz(
            radiance.photon_detection_rate(
                nm.snr.tgt_FPA_irradiance - nm.snr.other_irradiance,
                nm.sensor.w_x,
                nm.sensor.w_y,
                nm.radiance_wavelengths,
                nm.snr.qe,
            ),
            nm.radiance_wavelengths,
        )
        * nm.snr.int_time
        * sensor.n_tdi
    )
    bkg_noise = np.sqrt(
        np.trapz(
            radiance.photon_detection_rate(
                nm.snr.bkg_FPA_irradiance - nm.snr.other_irradiance,
                nm.sensor.w_x,
                nm.sensor.w_y,
                nm.radiance_wavelengths,
                nm.snr.qe,
            ),
            nm.radiance_wavelengths,
        )
        * nm.snr.int_time
        * sensor.n_tdi
    )  # assign the scene Noise to the larger of the target or background noise
    scene_and_path_noise = np.max([tgt_noise, bkg_noise])
    # calculate noise due to just the path scattered or emitted radiation
    scatter_rate, _, _ = radiance.signal_rate(
        nm.radiance_wavelengths,
        nm.atm[:, 2] + nm.atm[:, 4],
        nm.snr.opt_trans,
        nm.sensor.D,
        nm.sensor.f,
        nm.sensor.w_x,
        nm.sensor.w_y,
        nm.snr.qe,
        np.zeros(1),
        0.0,
    )
    nm.snr.path_noise = np.sqrt(scatter_rate * nm.snr.int_time * sensor.n_tdi)
    nm.snr.scene_noise = np.sqrt(scene_and_path_noise**2 - nm.snr.path_noise**2)
    # ######OTF CALCULATION#######

    # cut down the wavelength range to only the regions of interest
    nm.mtf_wavelengths = nm.radiance_wavelengths[nm.snr.weights > 0.0]
    nm.mtf_weights = nm.snr.weights[nm.snr.weights > 0.0]

    # setup spatial frequency array
    nm.cutoff_frequency = sensor.D / np.min(nm.mtf_wavelengths)
    u_rng = np.linspace(-1.0, 1.0, 101) * nm.cutoff_frequency
    v_rng = np.linspace(1.0, -1.0, 101) * nm.cutoff_frequency
    nm.uu, nm.vv = np.meshgrid(
        u_rng, v_rng
    )  # meshgrid of spatial frequencies out to the optics cutoff
    nm.df = u_rng[1] - u_rng[0]  # spatial frequency step size

    nm.otf = otf.functional.common_OTFs(
        sensor,
        scenario,
        nm.uu,
        nm.vv,
        nm.mtf_wavelengths,
        nm.mtf_weights,
        nm.slant_range,
        nm.snr.int_time,
    )

    # ########CALCULATE NIIRS##############
    nm.ifov_x = sensor.p_x / sensor.f
    nm.ifov_y = sensor.p_y / sensor.f
    nm.gsd_x = nm.ifov_x * nm.slant_range
    nm.gsd_y = nm.ifov_y * nm.slant_range
    nm.gsd_gm = np.sqrt(nm.gsd_x * nm.gsd_y)
    nm.rer_gm, nm.eho_gm = giqe_edge_terms(
        np.abs(nm.otf.system_OTF), nm.df, nm.ifov_x, nm.ifov_y
    )

    nm.ng = noise.noise_gain(sensor.filter_kernel)
    # note that NIIRS is calculated using the SNR ***after frame stacking****
    # if any
    nm.niirs = giqe3(
        nm.rer_gm,
        nm.gsd_gm,
        nm.eho_gm,
        nm.ng,
        np.sqrt(sensor.frame_stacks) * nm.snr.snr,
    )

    # NEW FOR VERSION 0.2 - GIQE 4
    nm.elev_angle = np.pi / 2 - geospatial.nadir_angle(
        0.0, scenario.altitude, nm.slant_range
    )
    nm.niirs_4, nm.gsd_gp = giqe4(
        nm.rer_gm,
        nm.gsd_gm,
        nm.eho_gm,
        nm.ng,
        np.sqrt(sensor.frame_stacks) * nm.snr.snr,
        nm.elev_angle,
    )
    # see pybsm.metrics.functional.niirs5 for GIQE 5

    return nm


def niirs5(sensor: Sensor, scenario: Scenario, interp: Optional[bool] = False) -> Metrics:
    """Returns NIIRS values calculate using GIQE 5 and all intermediate calculations.

    See pybsm.metrics.functional.niirs for the GIQE 3 version.  This version of the
    GIQE replaces the earlier versions and should be used in all future analyses.


    :param sensor:
        an object from the class sensor
    :param scenario:
        an object from the class scenario

    :return:
        nm:
            an object containing results of the GIQE calculation along with
            many intermediate calculations
    """
    # initialize the output
    nm = Metrics("niirs " + sensor.name + " " + scenario.name)
    nm.sensor = sensor
    nm.scenario = scenario
    nm.slant_range = geospatial.curved_earth_slant_range(
        0.0, scenario.altitude, scenario.ground_range
    )

    # #########CONTRAST SNR CALCULATION#########
    # load the atmosphere model
    if interp:
        nm.atm = utils.load_database_atmosphere(
            scenario.altitude, scenario.ground_range, scenario.ihaze
        )
    else:
        nm.atm = utils.load_database_atmosphere_no_interp(
            scenario.altitude, scenario.ground_range, scenario.ihaze
        )
    # crop out out-of-band data (saves time integrating later)
    nm.atm = nm.atm[nm.atm[:, 0] >= nm.sensor.opt_trans_wavelengths[0], :]
    nm.atm = nm.atm[nm.atm[:, 0] <= nm.sensor.opt_trans_wavelengths[-1], :]

    if nm.sensor.opt_trans_wavelengths[0] >= 2.9e-6:
        is_emissive = 1  # toggle the GIQE to assume infrared imaging
        # the next four lines are bookkeeping for interpreting the results
        # since the giqe_radiance function assumes these values anyway
        nm.scenario.target_reflectance = 0.0
        nm.scenario.background_reflectance = 0.0
        nm.scenario.target_temperature = 282.0
        nm.scenario.target_temperature = 280.0

    else:
        is_emissive = 0
        # more bookkeeping (see previous comment)
        nm.scenario.target_reflectance = 0.15
        nm.scenario.background_reflectance = 0.07

    # get aperture radiances for reflective target and background with GIQE
    # type target parameters
    nm.tgt_radiance, nm.bkg_radiance = radiance.giqe_radiance(nm.atm, is_emissive)
    nm.radiance_wavelengths = nm.atm[:, 0]

    # now calculate now characteristics ****for a single frame******
    nm.snr = radiance.photon_detector_SNR(
        sensor, nm.radiance_wavelengths, nm.tgt_radiance, nm.bkg_radiance
    )

    # break out photon noise sources (not required for NIIRS but useful for
    # analysis) photon noise due to the scene itself (target,background, and
    # path emissions/scattering)
    tgt_noise = np.sqrt(
        np.trapz(
            radiance.photon_detection_rate(
                nm.snr.tgt_FPA_irradiance - nm.snr.other_irradiance,
                nm.sensor.w_x,
                nm.sensor.w_y,
                nm.radiance_wavelengths,
                nm.snr.qe,
            ),
            nm.radiance_wavelengths,
        )
        * nm.snr.int_time
        * sensor.n_tdi
    )
    bkg_noise = np.sqrt(
        np.trapz(
            radiance.photon_detection_rate(
                nm.snr.bkg_FPA_irradiance - nm.snr.other_irradiance,
                nm.sensor.w_x,
                nm.sensor.w_y,
                nm.radiance_wavelengths,
                nm.snr.qe,
            ),
            nm.radiance_wavelengths,
        )
        * nm.snr.int_time
        * sensor.n_tdi
    )  # assign the scene Noise to the larger of the target or background noise
    scene_and_path_noise = np.max([tgt_noise, bkg_noise])
    # calculate noise due to just the path scattered or emitted radiation
    scatter_rate, _, _ = radiance.signal_rate(
        nm.radiance_wavelengths,
        nm.atm[:, 2] + nm.atm[:, 4],
        nm.snr.opt_trans,
        nm.sensor.D,
        nm.sensor.f,
        nm.sensor.w_x,
        nm.sensor.w_y,
        nm.snr.qe,
        np.zeros(1),
        0.0,
    )
    nm.snr.path_noise = np.sqrt(scatter_rate * nm.snr.int_time * sensor.n_tdi)
    nm.snr.scene_noise = np.sqrt(scene_and_path_noise**2 - nm.snr.path_noise**2)
    # ######OTF CALCULATION#######

    # cut down the wavelength range to only the regions of interest
    nm.mtf_wavelengths = nm.radiance_wavelengths[nm.snr.weights > 0.0]
    nm.mtf_weights = nm.snr.weights[nm.snr.weights > 0.0]

    # setup spatial frequency array
    nm.cutoff_frequency = sensor.D / np.min(nm.mtf_wavelengths)
    u_rng = np.linspace(-1.0, 1.0, 101) * nm.cutoff_frequency
    v_rng = np.linspace(1.0, -1.0, 101) * nm.cutoff_frequency
    nm.uu, nm.vv = np.meshgrid(
        u_rng, v_rng
    )  # meshgrid of spatial frequencies out to the optics cutoff
    nm.df = u_rng[1] - u_rng[0]  # spatial frequency step size

    sensor.filter_kernel = np.array(
        [1]
    )  # ensures that sharpening is turned off.  Not valid for GIQE5
    nm.otf = otf.functional.common_OTFs(
        sensor,
        scenario,
        nm.uu,
        nm.vv,
        nm.mtf_wavelengths,
        nm.mtf_weights,
        nm.slant_range,
        nm.snr.int_time,
    )

    # ##########CALCULATE NIIRS##############
    nm.ifov_x = sensor.p_x / sensor.f
    nm.ifov_y = sensor.p_y / sensor.f
    nm.gsd_x = nm.ifov_x * nm.slant_range  # GIQE5 assumes all square detectors
    nm.rer_0, nm.rer_90 = giqe5_RER(
        np.abs(nm.otf.system_OTF), nm.df, nm.ifov_x, nm.ifov_y
    )

    # note that NIIRS is calculated using the SNR ***after frame stacking****
    # if any
    nm.elev_angle = np.pi / 2 - geospatial.nadir_angle(
        0.0, scenario.altitude, nm.slant_range
    )
    nm.niirs, nm.gsd_w, nm.rer = giqe5(
        nm.rer_0,
        nm.rer_90,
        nm.gsd_x,
        np.sqrt(sensor.frame_stacks) * nm.snr.snr,
        nm.elev_angle,
    )

    return nm


def relative_edge_response(mtf_slice: np.ndarray, df: float, ifov: float) -> float:
    """IBSM Equation 3-61. The slope of the edge response of the system taken at +/-0.5 pixels from a theoretical edge.

    Edge response is used in the calculation of NIIRS via the General Image Quality Equation.

    :param mtf_slice:
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0
        cycles/radian
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifov:
        instantaneous field-of-view of a detector (radians)


    :return:
        rer:
            relative edge response (unitless)
    """
    rer = edge_response(0.5, mtf_slice, df, ifov) - edge_response(
        -0.5, mtf_slice, df, ifov
    )
    return rer


def edge_height_overshoot(mtf_slice: np.ndarray, df: float, ifov: float) -> float:
    """IBSM Equation 3-60.  Edge Height Overshoot is a measure of image distortion caused by sharpening.

    Note that there is a typo in Equation 3-60.  The correct definition is given in Leachtenauer et al.,
    "General Image-Quality Equation: GIQE" APPLIED OPTICS Vol. 36, No. 32 10 November 1997. "The
    overshoot-height term H models the edge response overshoot that is due to
    MTFC. It is measured over the range of 1.0 to 3.0 pixels from the edge
    in 0.25-pixel increments. If the edge is monotonically increasing, it is defined as the
    value at 1.25 pixels from the edge."

    :param mtf_slice:
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0
        cycles/radian
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifov:
        instantaneous field-of-view of a detector (radians)

    :return:
        eho :
            edge height overshoot (unitless)
    """
    rng = np.arange(1.0, 3.25, 0.25)
    er = np.zeros(rng.size)
    index = 0

    for dist in rng:
        er[index] = edge_response(dist, mtf_slice, df, ifov)
        index = index + 1

    if np.all(np.diff(er) > 0):  # when true, er is monotonically increasing
        eho = er[1]  # the edge response at 1.25 pixels from the edge
    else:
        eho = np.max(er)

    return eho


def edge_response(
    pixel_pos: float, mtf_slice: np.ndarray, df: float, ifov: float
) -> float:
    """IBSM Equation 3-63.  Imagine a perfectly sharp edge in object space.

    After the edge is blurred by the system MTF, the edge response is the normalized
    value of this blurred edge in image space at a distance of pixel_pos pixels
    away from the true edge.  Edge response is used in the calculation of NIIRS
    via the General Image Quality Equation.

    :param pixel_pos:
        distance from the theoretical edge (pixels)
    :param mtf_slice:
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0
        cycles/radian
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifov:
        instantaneous field-of-view of a detector (radians)


    :return:
        er:
            normalized edge response (unitless)
    """
    w = df * np.arange(1.0 * mtf_slice.size) + 1e-6  # note tiny offset to avoid infs
    y = (mtf_slice / w) * np.sin(2 * np.pi * w * ifov * pixel_pos)

    er = 0.5 + (1.0 / np.pi) * np.trapz(y, w)
    return er


def plot_common_MTFs(  # noqa: N802
    metrics: Metrics, orientation_angle: float = 0.0
) -> int:
    """Generates a plot of common MTF components.

    Generates a plot of common MTF components: aperture, turbulence, detector, jitter, drift, wavefront,
    image processing, system.  The Nyquist frequency is annotated on the plot with a black arrow.  Spatial
    frequencies are converted to image plane units (cycles/mm).

    :param metrics:
        the object output of (for instance) pybsm.metrics.functional.niirs or equivalent

    :param orientation_angle:
        angle to slice the MTF (radians).  A 0 radian slice is along the u
        axis. The angle rotates counterclockwise. Angle pi/2 is along the v
        axis. The default value is 0 radians.
    :return:
        a plot
    """
    # spatial frequencies in the image plane in (cycles/mm)
    rad_freq = np.sqrt(metrics.uu**2 + metrics.vv**2)
    sf = otf.functional.slice_otf(
        0.001 * (1.0 / metrics.sensor.f) * rad_freq, orientation_angle
    )

    # extract MTFs
    ap_mtf = np.abs(otf.functional.slice_otf(metrics.otf.ap_OTF, orientation_angle))
    turb_mtf = np.abs(otf.functional.slice_otf(metrics.otf.turb_OTF, orientation_angle))
    det_mtf = np.abs(otf.functional.slice_otf(metrics.otf.det_OTF, orientation_angle))
    jit_mtf = np.abs(otf.functional.slice_otf(metrics.otf.jit_OTF, orientation_angle))
    dri_mtf = np.abs(otf.functional.slice_otf(metrics.otf.drft_OTF, orientation_angle))
    wav_mtf = np.abs(otf.functional.slice_otf(metrics.otf.wav_OTF, orientation_angle))
    sys_mtf = np.abs(
        otf.functional.slice_otf(metrics.otf.system_OTF, orientation_angle)
    )
    fil_mtf = np.abs(
        otf.functional.slice_otf(metrics.otf.filter_OTF, orientation_angle)
    )

    plt.plot(
        sf,
        ap_mtf,
        sf,
        turb_mtf,
        sf,
        det_mtf,
        sf,
        jit_mtf,
        sf,
        dri_mtf,
        sf,
        wav_mtf,
        sf,
        fil_mtf,
        "gray",
    )
    plt.plot(sf, sys_mtf, "black", linewidth=2)
    plt.axis([0, sf.max(), 0, fil_mtf.max()])
    plt.xlabel("spatial frequency (cycles/mm)")
    plt.ylabel("MTF")
    plt.legend(
        [
            "aperture",
            "turbulence",
            "detector",
            "jitter",
            "drift",
            "wavefront",
            "image processing",
            "system",
        ]
    )

    # add nyquist frequency to plot
    nyquist = 0.5 / (metrics.ifov_x * metrics.sensor.f) * 0.001
    plt.annotate(
        "",
        xy=(nyquist, 0),
        xy_text=(nyquist, 0.1),
        arrow_props=dict(face_color="black", shrink=0.05),
    )
    return 0


def plot_noise_terms(metrics: Any, max_val: int = 0) -> int:
    """Generates a plot of common noise components in units of equivalent photoelectrons.

    Generates a plot of common noise components in units of equivalent photoelectrons: components total,
    scene photons, path photons, emission / stray photons, dark Current, quantization, readout.

    :param metrics:
        the object output of (for instance) pybsm.metrics.functional.niirs or equivalent

    :param max_val:
        (optional) sets the y-axis limit on photoelectrons.  Useful when
        comparing across plots. Default value of 0 allows matplotlib to
        automatically select the scale.
    :return:
        a plot
    """
    fig, ax = plt.subplots()
    ms = metrics.snr
    noise_terms = np.array(
        [
            ms.total_noise,
            ms.scene_noise,
            ms.path_noise,
            ms.self_emission_noise,
            ms.dark_current_noise,
            ms.quantization_noise,
            metrics.sensor.read_noise,
        ]
    )
    ind = np.arange(noise_terms.shape[0])
    ax.bar(ind, noise_terms, color="b")
    ax.set_ylabel("RMS Photoelectrons")
    ax.set_xticklabels(
        (
            " ",
            "Total",
            "Scene",
            "Path",
            "Emission / Stray",
            "Dark Current",
            "Quantization",
            "Readout",
        ),
        rotation=45,
    )
    # NOTE: I'm not sure why the first label is ingnored in the previous line
    # of code seems to be a new issue when I transitions to Python 3.6
    plt.title(
        "total photoelectrons per pixel: "
        + str(int(metrics.snr.tgt_n))
        + "\n contrast photoelectrons per pixel: "
        + str(int(metrics.snr.tgt_n - metrics.snr.bkg_n))
        + "\n well fill: "
        + str(int(metrics.snr.well_fraction * 100.0))
        + "%"
    )
    if max_val > 0:
        plt.ylim([0, max_val])
    plt.tight_layout()  # prevents text from getting cutoff
    return 0
