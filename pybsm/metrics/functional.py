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
"""

# standard library imports
import os
import inspect
import warnings

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple

# local imports
import pybsm.otf as otf
import pybsm.radiance as radiance
from pybsm import geospatial
from pybsm import noise
from pybsm import utils
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
# dirpath = os.path.dirname(os.path.abspath(__file__))
dirpath = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def giqe3(
    rer: float,
    gsd: float,
    eho: float,
    ng: float,
    snr: float
) -> float:
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
    niirs = (
        11.81 + 3.32 * np.log10(rer / (gsd / 0.0254)) - 1.48 * eho - ng / snr
    )
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion
    return niirs


def giqe4(
    rer: float,
    gsd: float,
    eho: float,
    ng: float,
    snr: float,
    elevAngle: float
) -> Tuple[float, float]:
    """General Image Quality Equation version 4 from Leachtenauer, et al.,
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
    :param elevangle:
        sensor elevation angle as measured from the target (rad), i.e.
        pi/2-nadirAngle. See pybsm.geospatial.nadirAngle for more information. Note that
        the GIQE4 paper defines this angle differently but we stick with this
        version to be consistent with the GIQE 5 code. The outcome is the same
        either way.

    :return:
        niirs :
            a National Image Interpretability Rating Scale value (unitless)
    """
    if rer >= 0.9:
        c1 = 3.32
        c2 = 1.559
    else:
        c1 = 3.16
        c2 = 2.817

    gsdgp = gsd / (
        np.sin(elevAngle) ** (0.5)
    )  # note that the exponent captures the
    # fact that only one direction in the gsd is distorted by projection into
    # the ground plane

    niirs = (
        10.251
        - c1 * np.log10(gsdgp / 0.0254)
        + c2 * np.log10(rer)
        - 0.656 * eho
        - 0.334 * ng / snr
    )
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion
    return niirs, gsdgp


def giqe5(
    rer1: float,
    rer2: float,
    gsd: float,
    snr: float,
    elevAngle: float
) -> Tuple[float, float, float]:
    """NGA The General Image Quality Equation version 5.0. 16 Sep 2015
    https://gwg.nga.mil/ntb/baseline/docs/GIQE-5_for_Public_Release.pdf
    This version of the GIQE replaces the earlier versions and should be used
    in all future analyses.  See also "Airborne Validation of the General Image
    Quality Equation 5"
    https://www.osapublishing.org/ao/abstract.cfm?uri=ao-59-32-9978

    :param rer1:
        relative edge response in two directions (e.g., along- and across-
        scan, horizontal and vertical, etc.); see also pybsm.metrics.functional.giqe5RER.
        (unitless)
    :param rer2:
        relative edge response in two directions (e.g., along- and across-
        scan, horizontal and vertical, etc.); see also pybsm.metrics.functional.giqe5RER.
        (unitless)
    :param gsd:
        image plane geometric mean ground sample distance (m), as defined for
        GIQE3; the GIQE 5 version of GSD is calculated within this function
    :param snr:
        contrast signal-to-noise ratio (unitless), as defined for GIQE3
    :param elevangle:
        sensor elevation angle as measured from the target (rad), i.e.
        pi/2-nadirAngle; see pybsm.geospatial.nadirAngle for more information

    :return:
        niirs :
            a National Image Interpretability Rating Scale value (unitless)
        gsdw :
            elevation angle weighted GSD (m)
        rer :
            weighted relative edge response (rer)
    :NOTE:
        NIIRS 5 test case: rer1=rer2=0.35, gsd = 0.52832 (20.8 inches),
        snr = 50, elevangle = np.pi/2. From Figure 1 in the NGA GIQE5 paper.

    """
    # note that, within the GIQE, gsd is defined in inches, hence the
    # conversion in the niirs equation below
    gsdw = gsd / (
        np.sin(elevAngle) ** (0.25)
    )  # geometric mean of the image plane and ground plane gsds

    rer = (np.max([rer1, rer2]) * np.min([rer1, rer2]) ** 2.0) ** (1.0 / 3.0)

    niirs = (
        9.57
        - 3.32 * np.log10(gsdw / 0.0254)
        + 3.32 * (1 - np.exp(-1.9 / snr)) * np.log10(rer)
        - 2.0 * np.log10(rer) ** 4.0
        - 1.8 / snr
    )
    return niirs, gsdw, rer


def giqe5RER(
    mtf: np.ndarray,
    df: float,
    ifovx: float,
    ifovy: float
) -> Tuple[float, float]:
    """Calculates the relative edge response from a 2-D MTF.  This function is
    primarily for use with the GIQE 5. It implements IBSM equations 3-57 and
    3-58.  See pybsm.metrics.functional.giqeEdgeTerms for the GIQE 3 version.

    :param mtf:
        2-dimensional full system modulation transfer function (unitless);
        MTF is the magnitude of the OTF
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifovx:
        x-direction instantaneous field-of-view of a detector (radians)
    :param ifovy:
        y-direction instantaneous field-of-view of a detector (radians)

    :return:
        rer0:
            relative edge response at 0 degrees orientation (unitless)
        rer90:
            relative edge response at 90 degrees orientation (unitless)
    """
    uslice = otf.functional.sliceotf(mtf, 0)
    vslice = otf.functional.sliceotf(mtf, np.pi / 2)

    rer0 = relativeEdgeResponse(uslice, df, ifovx)
    rer90 = relativeEdgeResponse(vslice, df, ifovy)

    return rer0, rer90


def giqeEdgeTerms(
    mtf: np.ndarray,
    df: float,
    ifovx: float,
    ifovy: float
) -> Tuple[float, float]:
    """Calculates the geometric mean relative edge response and edge high overshoot,
    from a 2-D MTF.  This function is primarily for use with the GIQE. It
    implements IBSM equations 3-57 and 3-58.

    :param mtf:
        2-dimensional full system modulation transfer function (unitless);
        MTF is the magnitude of the OTF
    :param df:
        spatial frequency step size (cycles/radian)
    :param ifovx:
        x-direction instantaneous field-of-view of a detector (radians)
    :param ifovy:
        y-direction instantaneous field-of-view of a detector (radians)

    :return:
        rer:
            geometric mean relative edge response (unitless)
        eho:
            geometric mean edge height overshoot (unitless)
    """
    uslice = otf.functional.sliceotf(mtf, 0)
    vslice = otf.functional.sliceotf(mtf, np.pi / 2)

    urer = relativeEdgeResponse(uslice, df, ifovx)
    vrer = relativeEdgeResponse(vslice, df, ifovy)
    rer = np.sqrt(urer * vrer)

    ueho = edgeHeightOvershoot(uslice, df, ifovx)
    veho = edgeHeightOvershoot(vslice, df, ifovy)
    eho = np.sqrt(ueho * veho)

    return rer, eho


def groundResolvedDistance(
    mtfslice: np.ndarray,
    df: float,
    snr: float,
    ifov: float,
    slantRange: float
) -> float:
    """IBSM Equation 3-54.  The ground resolved distance is the period of the
    smallest square wave pattern that can be resolved in an image.  GRD can be
    limited by the detector itself (in which case GRD = 2*GSD) but, in general
    is a function of the system MTF and signal-to-noise ratio.

    :param mtfslice:
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0
        cycles/radian
    :param df:
        spatial frequency step size (cycles/radian)
    :param snr:
        contrast signal-to-noise ratio (unitless)
    :param ifov:
        instantaneous field-of-view of a detector (radians)
    :param slantRange:
        distance between the target and sensor (m)
    :return:
        grd:
            ground resolved distance (m)
    """
    w = df * np.arange(1.0 * mtfslice.size)
    ur = (
        np.interp(3.0 / snr, mtfslice[::-1], w[::-1]) + 1e-12
    )  # 1e-12 prevents division by zero in grdcases
    # arrays were reversed to satisfy the requirements of np.interp

    grdcases = slantRange * np.array([1.0 / ur, 2.0 * ifov])
    grd = np.max(grdcases)

    return grd


def niirs(
    sensor: Sensor,
    scenario: Scenario
) -> Metrics:
    """Returns NIIRS values and all intermediate calculations.  This function
    implements the original MATLAB-based NIIRS model and can serve as a
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
    nm.slantRange = geospatial.curvedEarthSlantRange(
        0.0, scenario.altitude, scenario.ground_range
    )

    # #########CONTRAST SNR CALCULATION#########
    # load the atmosphere model
    nm.atm = utils.loadDatabaseAtmosphere(
        scenario.altitude, scenario.ground_range, scenario.ihaze
    )

    # crop out out-of-band data (saves time integrating later)
    nm.atm = nm.atm[nm.atm[:, 0] >= nm.sensor.optTransWavelengths[0], :]
    nm.atm = nm.atm[nm.atm[:, 0] <= nm.sensor.optTransWavelengths[-1], :]

    if nm.sensor.optTransWavelengths[0] >= 2.9e-6:
        isEmissive = 1  # toggle the GIQE to assume infrared imaging
        # the next four lines are bookkeeping for interpreting the results
        # since the giqeRadiance function assumes these values anyway
        nm.scenario.targetReflectance = 0.0
        nm.scenario.backgroundReflectance = 0.0
        nm.scenario.targetTemperature = 282.0
        nm.scenario.targetTemperature = 280.0

    else:
        isEmissive = 0
        # more bookkeeping (see previous comment)
        nm.scenario.targetReflectance = 0.15
        nm.scenario.backgroundReflectance = 0.07

    # get aperture radiances for reflective target and background with GIQE
    # type target parameters
    nm.tgtRadiance, nm.bkgRadiance = radiance.giqeRadiance(nm.atm, isEmissive)
    nm.radianceWavelengths = nm.atm[:, 0]

    # now calculate now characteristics ****for a single frame******
    nm.snr = radiance.photonDetectorSNR(
        sensor, nm.radianceWavelengths, nm.tgtRadiance, nm.bkgRadiance
    )

    # break out photon noise sources (not required for NIIRS but useful for
    # analysis) photon noise due to the scene itself (target,background, and
    # path emissions/scattering)
    tgtNoise = np.sqrt(
        np.trapz(
            radiance.photonDetectionRate(
                nm.snr.tgtFPAirradiance - nm.snr.otherIrradiance,
                nm.sensor.wx,
                nm.sensor.wy,
                nm.radianceWavelengths,
                nm.snr.qe,
            ),
            nm.radianceWavelengths,
        )
        * nm.snr.intTime
        * sensor.ntdi
    )
    bkgNoise = np.sqrt(
        np.trapz(
            radiance.photonDetectionRate(
                nm.snr.bkgFPAirradiance - nm.snr.otherIrradiance,
                nm.sensor.wx,
                nm.sensor.wy,
                nm.radianceWavelengths,
                nm.snr.qe,
            ),
            nm.radianceWavelengths,
        )
        * nm.snr.intTime
        * sensor.ntdi
    )  # assign the scene Noise to the larger of the target or background noise
    sceneAndPathNoise = np.max([tgtNoise, bkgNoise])
    # calculate noise due to just the path scattered or emitted radiation
    scatrate, _, _ = radiance.signalRate(
        nm.radianceWavelengths,
        nm.atm[:, 2] + nm.atm[:, 4],
        nm.snr.optTrans,
        nm.sensor.D,
        nm.sensor.f,
        nm.sensor.wx,
        nm.sensor.wy,
        nm.snr.qe,
        np.zeros(1),
        0.0,
    )
    nm.snr.pathNoise = np.sqrt(scatrate * nm.snr.intTime * sensor.ntdi)
    nm.snr.sceneNoise = np.sqrt(sceneAndPathNoise**2 - nm.snr.pathNoise**2)
    # ######OTF CALCULATION#######

    # cut down the wavelength range to only the regions of interest
    nm.mtfwavelengths = nm.radianceWavelengths[nm.snr.weights > 0.0]
    nm.mtfweights = nm.snr.weights[nm.snr.weights > 0.0]

    # setup spatial frequency array
    nm.cutoffFrequency = sensor.D / np.min(nm.mtfwavelengths)
    urng = np.linspace(-1.0, 1.0, 101) * nm.cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 101) * nm.cutoffFrequency
    nm.uu, nm.vv = np.meshgrid(
        urng, vrng
    )  # meshgrid of spatial frequencies out to the optics cutoff
    nm.df = urng[1] - urng[0]  # spatial frequency step size

    nm.otf = otf.functional.commonOTFs(
        sensor,
        scenario,
        nm.uu,
        nm.vv,
        nm.mtfwavelengths,
        nm.mtfweights,
        nm.slantRange,
        nm.snr.intTime,
    )

    # ########CALCULATE NIIRS##############
    nm.ifovx = sensor.px / sensor.f
    nm.ifovy = sensor.py / sensor.f
    nm.gsdx = nm.ifovx * nm.slantRange
    nm.gsdy = nm.ifovy * nm.slantRange
    nm.gsdgm = np.sqrt(nm.gsdx * nm.gsdy)
    nm.rergm, nm.ehogm = giqeEdgeTerms(
        np.abs(nm.otf.systemOTF), nm.df, nm.ifovx, nm.ifovy
    )

    nm.ng = noise.noiseGain(sensor.filterKernel)
    # note that NIIRS is calculated using the SNR ***after frame stacking****
    # if any
    nm.niirs = giqe3(
        nm.rergm,
        nm.gsdgm,
        nm.ehogm,
        nm.ng,
        np.sqrt(sensor.framestacks) * nm.snr.snr,
    )

    # NEW FOR VERSION 0.2 - GIQE 4
    nm.elevAngle = np.pi / 2 - geospatial.nadirAngle(
        0.0, scenario.altitude, nm.slantRange
    )
    nm.niirs4, nm.gsdgp = giqe4(
        nm.rergm,
        nm.gsdgm,
        nm.ehogm,
        nm.ng,
        np.sqrt(sensor.framestacks) * nm.snr.snr,
        nm.elevAngle,
    )
    # see pybsm.metrics.functional.niir5 for GIQE 5

    return nm


def niirs5(
    sensor: Sensor,
    scenario: Scenario
) -> Metrics:
    """Returns NIIRS values calculate using GIQE 5 and all intermediate
    calculations.  See pybsm.metrics.functional.niirs for the GIQE 3 version.  This version of the
    GIQE replaces the earlier versions and should be used in all future
    analyses.


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
    nm.slantRange = geospatial.curvedEarthSlantRange(
        0.0, scenario.altitude, scenario.ground_range
    )

    # #########CONTRAST SNR CALCULATION#########
    # load the atmosphere model
    nm.atm = utils.loadDatabaseAtmosphere(
        scenario.altitude, scenario.ground_range, scenario.ihaze
    )

    # crop out out-of-band data (saves time integrating later)
    nm.atm = nm.atm[nm.atm[:, 0] >= nm.sensor.optTransWavelengths[0], :]
    nm.atm = nm.atm[nm.atm[:, 0] <= nm.sensor.optTransWavelengths[-1], :]

    if nm.sensor.optTransWavelengths[0] >= 2.9e-6:
        isEmissive = 1  # toggle the GIQE to assume infrared imaging
        # the next four lines are bookkeeping for interpreting the results
        # since the giqeRadiance function assumes these values anyway
        nm.scenario.targetReflectance = 0.0
        nm.scenario.backgroundReflectance = 0.0
        nm.scenario.targetTemperature = 282.0
        nm.scenario.targetTemperature = 280.0

    else:
        isEmissive = 0
        # more bookkeeping (see previous comment)
        nm.scenario.targetReflectance = 0.15
        nm.scenario.backgroundReflectance = 0.07

    # get aperture radiances for reflective target and background with GIQE
    # type target parameters
    nm.tgtRadiance, nm.bkgRadiance = radiance.giqeRadiance(nm.atm, isEmissive)
    nm.radianceWavelengths = nm.atm[:, 0]

    # now calculate now characteristics ****for a single frame******
    nm.snr = radiance.photonDetectorSNR(
        sensor, nm.radianceWavelengths, nm.tgtRadiance, nm.bkgRadiance
    )

    # break out photon noise sources (not required for NIIRS but useful for
    # analysis) photon noise due to the scene itself (target,background, and
    # path emissions/scattering)
    tgtNoise = np.sqrt(
        np.trapz(
            radiance.photonDetectionRate(
                nm.snr.tgtFPAirradiance - nm.snr.otherIrradiance,
                nm.sensor.wx,
                nm.sensor.wy,
                nm.radianceWavelengths,
                nm.snr.qe,
            ),
            nm.radianceWavelengths,
        )
        * nm.snr.intTime
        * sensor.ntdi
    )
    bkgNoise = np.sqrt(
        np.trapz(
            radiance.photonDetectionRate(
                nm.snr.bkgFPAirradiance - nm.snr.otherIrradiance,
                nm.sensor.wx,
                nm.sensor.wy,
                nm.radianceWavelengths,
                nm.snr.qe,
            ),
            nm.radianceWavelengths,
        )
        * nm.snr.intTime
        * sensor.ntdi
    )  # assign the scene Noise to the larger of the target or background noise
    sceneAndPathNoise = np.max([tgtNoise, bkgNoise])
    # calculate noise due to just the path scattered or emitted radiation
    scatrate, _, _ = radiance.signalRate(
        nm.radianceWavelengths,
        nm.atm[:, 2] + nm.atm[:, 4],
        nm.snr.optTrans,
        nm.sensor.D,
        nm.sensor.f,
        nm.sensor.wx,
        nm.sensor.wy,
        nm.snr.qe,
        np.zeros(1),
        0.0,
    )
    nm.snr.pathNoise = np.sqrt(scatrate * nm.snr.intTime * sensor.ntdi)
    nm.snr.sceneNoise = np.sqrt(sceneAndPathNoise**2 - nm.snr.pathNoise**2)
    # ######OTF CALCULATION#######

    # cut down the wavelength range to only the regions of interest
    nm.mtfwavelengths = nm.radianceWavelengths[nm.snr.weights > 0.0]
    nm.mtfweights = nm.snr.weights[nm.snr.weights > 0.0]

    # setup spatial frequency array
    nm.cutoffFrequency = sensor.D / np.min(nm.mtfwavelengths)
    urng = np.linspace(-1.0, 1.0, 101) * nm.cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 101) * nm.cutoffFrequency
    nm.uu, nm.vv = np.meshgrid(
        urng, vrng
    )  # meshgrid of spatial frequencies out to the optics cutoff
    nm.df = urng[1] - urng[0]  # spatial frequency step size

    sensor.filterKernel = np.array(
        [1]
    )  # ensures that sharpening is turned off.  Not valid for GIQE5
    nm.otf = otf.functional.commonOTFs(
        sensor,
        scenario,
        nm.uu,
        nm.vv,
        nm.mtfwavelengths,
        nm.mtfweights,
        nm.slantRange,
        nm.snr.intTime,
    )

    # ##########CALCULATE NIIRS##############
    nm.ifovx = sensor.px / sensor.f
    nm.ifovy = sensor.py / sensor.f
    nm.gsdx = nm.ifovx * nm.slantRange  # GIQE5 assumes all square detectors
    nm.rer0, nm.rer90 = giqe5RER(
        np.abs(nm.otf.systemOTF), nm.df, nm.ifovx, nm.ifovy
    )

    # note that NIIRS is calculated using the SNR ***after frame stacking****
    # if any
    nm.elevAngle = np.pi / 2 - geospatial.nadirAngle(
        0.0, scenario.altitude, nm.slantRange
    )
    nm.niirs, nm.gsdw, nm.rer = giqe5(
        nm.rer0,
        nm.rer90,
        nm.gsdx,
        np.sqrt(sensor.framestacks) * nm.snr.snr,
        nm.elevAngle,
    )

    return nm


def relativeEdgeResponse(
    mtfslice: np.ndarray,
    df: float,
    ifov: float
) -> float:
    """IBSM Equation 3-61.  The slope of the edge response of the system taken
    at +/-0.5 pixels from a theoretical edge.  Edge response is used in the
    calculation of NIIRS via the General Image Quality Equation.

    :param mtfslice:
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
    rer = edgeResponse(0.5, mtfslice, df, ifov) - edgeResponse(
        -0.5, mtfslice, df, ifov
    )
    return rer


def edgeHeightOvershoot(
    mtfslice: np.ndarray,
    df: float,
    ifov: float
) -> float:
    """IBSM Equation 3-60.  Edge Height Overshoot is a measure of image distortion
    caused by sharpening.  Note that there is a typo in Equation 3-60.  The
    correct definition is given in Leachtenauer et al., "General Image-Quality
    Equation: GIQE" APPLIED OPTICS Vol. 36, No. 32 10 November 1997. "The
    overshoot-height term H models the edge response overshoot that is due to
    MTFC. It is measured over the range of 1.0 to 3.0 pixels from the edge
    in 0.25-pixel increments. If the edge is monotonically increasing, it is defined as the
    value at 1.25 pixels from the edge."

    :param mtfslice:
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
        er[index] = edgeResponse(dist, mtfslice, df, ifov)
        index = index + 1

    if np.all(np.diff(er) > 0):  # when true, er is monotonically increasing
        eho = er[1]  # the edge response at 1.25 pixels from the edge
    else:
        eho = np.max(er)

    return eho


def edgeResponse(
    pixelPos: float,
    mtfslice: np.ndarray,
    df: float,
    ifov: float
) -> float:
    """IBSM Equation 3-63.  Imagine a perfectly sharp edge in object space.  After
    the edge is blurred by the system MTF, the edge response is the normalized
    value of this blurred edge in image space at a distance of pixelPos pixels
    away from the true edge.  Edge response is used in the calculation of NIIRS
    via the General Image Quality Equation.

    :param pixelPos:
        distance from the theoretical edge (pixels)
    :param mtfslice:
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
    w = (
        df * np.arange(1.0 * mtfslice.size) + 1e-6
    )  # note tiny offset to avoid infs
    y = (mtfslice / w) * np.sin(2 * np.pi * w * ifov * pixelPos)

    er = 0.5 + (1.0 / np.pi) * np.trapz(y, w)
    return er


def plotCommonMTFs(
    metrics: Metrics,
    orientationAngle: float = 0.0
) -> int:
    """Generates a plot of common MTF components: aperture, turbulence, detector,
    jitter, drift, wavefront, image processing, system.  The Nyquist frequency is
    annotated on the plot with a black arrow.  Spatial frequencies are
    converted to image plane units (cycles/mm).

    :param metrics:
        the object output of (for instance) pybsm.metrics.functional.niirs or equivalent

    :param orientationAngle:
        angle to slice the MTF (radians).  A 0 radian slice is along the u
        axis. The angle rotates counterclockwise. Angle pi/2 is along the v
        axis. The default value is 0 radians.
    :return:
        a plot
    """

    # spatial frequencies in the image plane in (cycles/mm)
    radfreq = np.sqrt(metrics.uu**2 + metrics.vv**2)
    sf = otf.functional.sliceotf(
        0.001 * (1.0 / metrics.sensor.f) * radfreq, orientationAngle
    )

    # extract MTFs
    apmtf = np.abs(otf.functional.sliceotf(metrics.otf.apOTF, orientationAngle))
    turbmtf = np.abs(otf.functional.sliceotf(metrics.otf.turbOTF, orientationAngle))
    detmtf = np.abs(otf.functional.sliceotf(metrics.otf.detOTF, orientationAngle))
    jitmtf = np.abs(otf.functional.sliceotf(metrics.otf.jitOTF, orientationAngle))
    drimtf = np.abs(otf.functional.sliceotf(metrics.otf.drftOTF, orientationAngle))
    wavmtf = np.abs(otf.functional.sliceotf(metrics.otf.wavOTF, orientationAngle))
    sysmtf = np.abs(otf.functional.sliceotf(metrics.otf.systemOTF, orientationAngle))
    filmtf = np.abs(otf.functional.sliceotf(metrics.otf.filterOTF, orientationAngle))

    plt.plot(
        sf,
        apmtf,
        sf,
        turbmtf,
        sf,
        detmtf,
        sf,
        jitmtf,
        sf,
        drimtf,
        sf,
        wavmtf,
        sf,
        filmtf,
        "gray",
    )
    plt.plot(sf, sysmtf, "black", linewidth=2)
    plt.axis([0, sf.max(), 0, filmtf.max()])
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
    nyquist = 0.5 / (metrics.ifovx * metrics.sensor.f) * 0.001
    plt.annotate(
        "",
        xy=(nyquist, 0),
        xytext=(nyquist, 0.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    return 0


def plotNoiseTerms(
    metrics: Any,
    maxval: int = 0
) -> int:
    """Generates a plot of common noise components in units of equivalent
    photoelectrons: components total, scene photons, path photons, emission /
    stray photons, dark Current, quantization, readout.

    :param metrics:
        the object output of (for instance) pybsm.metrics.functional.niirs or equivalent

    :param maxval:
        (optional) sets the y-axis limit on photoelectrons.  Useful when
        comparing across plots. Default value of 0 allows matplotlib to
        automatically select the scale.
    :return:
        a plot
    """
    fig, ax = plt.subplots()
    ms = metrics.snr
    noiseterms = np.array(
        [
            ms.totalNoise,
            ms.sceneNoise,
            ms.pathNoise,
            ms.selfEmissionNoise,
            ms.darkcurrentNoise,
            ms.quantizationNoise,
            metrics.sensor.readNoise,
        ]
    )
    ind = np.arange(noiseterms.shape[0])
    ax.bar(ind, noiseterms, color="b")
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
        + str(int(metrics.snr.tgtN))
        + "\n contrast photoelectrons per pixel: "
        + str(int(metrics.snr.tgtN - metrics.snr.bkgN))
        + "\n well fill: "
        + str(int(metrics.snr.wellfraction * 100.0))
        + "%"
    )
    if maxval > 0:
        plt.ylim([0, maxval])
    plt.tight_layout()  # prevents text from getting cutoff
    return 0
