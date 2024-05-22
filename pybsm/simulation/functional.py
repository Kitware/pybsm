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
import logging
import inspect
import warnings

# 3rd party imports
import numpy as np
from scipy import interpolate
from typing import List, Tuple

# local imports
import pybsm.otf as otf
import pybsm.radiance as radiance
from pybsm import noise
from .sensor import Sensor
from .scenario import Scenario
from .ref_image import RefImage

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


def instantaneousFOV(
    w: int,
    f: int
) -> float:
    """The instantaneous field of view; i.e., the angular footprint of a single
    detector in object space.

    :param w:
        detector size (width) in the x and y directions (m)
    :type w: int
    :param f:
        focal length (m)
    :type f: int

    :return:
        ifov:
            detector instantaneous field-of-view (radians)
    """
    ifov = w / f
    return ifov


def wienerFilter(
    otf: np.ndarray,
    noiseToSignalPS: float
) -> np.ndarray:
    """An image restoration transfer function based on the Wiener Filter.  See
    from Gonzalex and Woods, "Digital Image Processing," 3 ed.  Note that the
    filter is not normalized so that WF = 1 at 0 spatial frequency.  This is
    easily fixed for the case where noiseToSignalPS is a
    scalar: (1.0+noisetosignalPS)*WF = 1 at 0 spatial frequency.
    This is noteworthy because, without normalization of some kind, system
    MTF * WF is not a proper MTF. Also, for any frequency space sharpening
    kernel, be sure that sharpening does not extend past the Nyquist frequency,
    or, if so, ensure that the filter response wraps around Nyquist
    appropriately.

    :param otf:
        system optical transfer function
    :type otf: list
    :param noiseTosignalPS:
        ratio of the noise power spectrum to the signal power spectrum; this
        may be a function of spatial frequency (same size as otf) or a scalar
    :type noiseTosignalPS: float

    :return:
        WF:
            frequency space representation of the Wienerfilter
    """
    WF = np.conj(otf) / (np.abs(otf) ** 2 + noiseToSignalPS)

    return WF


def img2reflectance(
    img: np.ndarray,
    pix_values: np.ndarray,
    refl_values: np.ndarray
) -> np.ndarray:
    """Maps pixel values to reflectance values with linear interpolation
    between points. Pixel values that map below zero reflectance or above
    unity reflectance are truncated. Implicitly, reflectance is contrast
    across the camera bandpass.

    :param img:
        the image that will be transformed into reflectance (counts)
    :type img: np.array
    :param pix_values:
        array of values in img that map to a known reflectance (counts)
    :type pix_values: np.array
    :param refl_values:
        array of reflectances that correspond to pix_values (unitless)
    :type refl_values: np.array

    :return:
        refImg:
            the image in reflectance space

    :raises:
        ValueError:
            if img, pix_values, or refl_values have a length < 2

    :WARNING:
        Output can be nan if all input arrays have the same values.
    """
    f = interpolate.interp1d(
        pix_values,
        refl_values,
        fill_value="extrapolate",
        assume_sorted=0,
    )
    refImg = f(img)
    refImg[refImg > 1.0] = 1.0
    refImg[refImg < 0.0] = 0.0

    return refImg


def simulate_image(
    ref_img: RefImage,
    sensor: Sensor,
    scenario: Scenario
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulates radiometrically accurate imagery collected through a sensor.

    We start with a notionally ideal reference image 'img', which captures a
    view of the world of which we would like to simulate a degraded view that
    would be collected from another imaging system. Our framework for image
    simulation requires that this reference image is of higher quality and
    ideally higher resolution than the view we would like to simulate as we can
    only model further degradation of image quality.


    :type ref_img: RefImage
    :param ref_img:
        Reference image to use as the source view of the world to be emulated
        by the virtual camera defined by 'sensor'.
    :type sensor: Sensor
    :param sensor:
        Virtual sensor definition.
    :type scenario: Scenario
    :param scenario:
        Specification of the deployment of the virtual sensor within the world
        relative to the target.

    :return:
        trueImg:
            Numpy float64 array; the true image in units of photoelectrons
        blurImg:
            Numpy float64 array; the image after blurring and resampling is applied to trueImg
            (still units of photoelectrons)
        noisyImg:
            Numpy float64 array; the blur image with photon (Poisson) noise and gaussian noise
            applied (still units of photoelectrons)

    :WARNING:
        imggsd must be small enough to properly sample the blur kernel. As a
        guide, if the image system transfer function goes to zero at angular
        spatial frequency, coff, then the sampling requirement will be readily
        met if imggsd <= rng/(4*coff). In practice this is easily done by
        upsampling imgin.

    :raises: ValueError if cutoff Frequency matrix urng is not monotonically
             increasing
    """
    # integration time (s)
    intTime = sensor.intTime

    (
        ref,
        pe,
        spectral_weights,
    ) = radiance.reflectance2photoelectrons(scenario.atm, sensor, intTime)

    wavelengths = spectral_weights[0]
    weights = spectral_weights[1]

    slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)

    # cut down the wavelength range to only the regions of interest
    mtfwavelengths = wavelengths[weights > 0.0]
    mtfweights = weights[weights > 0.0]

    # Assume if nothing else cuts us off first, diffraction will set the limit
    # for spatial frequency that the imaging system can resolve (1/rad).
    cutoffFrequency = sensor.D / np.min(mtfwavelengths)

    urng = np.linspace(-1.0, 1.0, 1501) * cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 1501) * cutoffFrequency

    # meshgrid of spatial frequencies out to the optics cutoff
    uu, vv = np.meshgrid(urng, vrng)

    system_otf = otf.commonOTFs(
        sensor,
        scenario,
        uu,
        vv,
        mtfwavelengths,
        mtfweights,
        slant_range,
        intTime,
    ).systemOTF

    df = (abs(urng[1] - urng[0]) + abs(vrng[0] - vrng[1]))/2

    if df <= 0:
        raise ValueError("Cutoff frequency values must be increasing.")

    ifov = (sensor.px + sensor.py) / 2 / sensor.f

    # Standard deviation of additive Gaussian noise (e.g. read noise,
    # quantization). Should be the RSS value if multiple terms are combined.
    # This should not include photon noise.
    quantizationNoise = noise.quantizationNoise(sensor.maxN, sensor.bitdepth)
    gnoise = np.sqrt(quantizationNoise**2.0 + sensor.readNoise**2.0)

    # Convert to reference image into a floating point reflectance image.
    refImg = img2reflectance(
        ref_img.img, ref_img.pix_values, ref_img.refl_values
    )

    # Convert the reflectance image to photoelectrons.
    f = interpolate.interp1d(ref, pe)
    trueImg = f(refImg)

    # blur and resample the image
    blurImg, _ = otf.apply_otf_to_image(
        trueImg, ref_img.gsd, slant_range, system_otf, df, ifov
    )

    # add photon noise (all sources) and dark current noise
    poisson_noisy_img = np.random.poisson(lam=blurImg)
    # add any noise from Gaussian sources, e.g. readnoise, quantizaiton
    noisyImg = np.random.normal(poisson_noisy_img, gnoise)

    if noisyImg.shape[0] > ref_img.img.shape[0]:
        logging.warn(
            "The simulated image has oversampled the"
            " reference image!  This result should not be"
            " trusted!!"
        )

    return trueImg, blurImg, noisyImg


def stretch_contrast_convert_8bit(
    img: np.ndarray,
    perc: List[float] = [0.1, 99.9]
) -> np.ndarray:
    img = img.astype(float)
    img = img - np.percentile(img.ravel(), perc[0])
    img = img / (np.percentile(img.ravel(), perc[1]) / 255)
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)
