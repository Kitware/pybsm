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

This module deals with all things related to radiance, including black body
spectral emmission calculations, spectral atmospheric absortion, and conversion
of radiance to photons. Anything material with a non-zero temperature emmits
electromagnetic radiation according to Planck's law and its tempreature,
spectral emissivity.
"""
# standard library imports
import os
import inspect
import warnings

# 3rd party imports
import numpy as np

# local imports
from pybsm import metrics
from pybsm import noise

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


def atFocalPlaneIrradiance(D, f, L):
    """Converts pupil plane radiance to focal plane irradiance for an extended
    source.This is a variation on part of IBSM Equation 3-34.  There is one
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
        E: total irradiance (W/m^2) or spectral irradiance (W/m^2 m) at the
        focal plane

    """
    E = L * np.pi / (1.0 + 4.0 * (f / D) ** 2.0)

    return E


def blackbodyRadiance(lambda0, T):
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
    Lbb = (2.0 * hc * cc**2.0 / lambda0**5.0) * (
        np.exp(hc * cc / (lambda0 * kc * T)) - 1.0
    ) ** (-1.0)

    return Lbb


def checkWellFill(totalPhotoelectrons, maxfill):
    """Check to see if the total collected photoelectrons are greater than the
    desired maximum well fill.  If so, provide a scale factor to reduce the
    integration time.

    :param totalPhotoelectrons:
        array of wavelengths (m)
    :type totalPhotelectrons: np.array
    :param maxFill:
        desired well fill, i.e. Maximum well size x Desired fill fraction
    :type maxFill: float

    :return:
        scalefactor:
            the new integration time is scaled by scalefactor
    """
    scalefactor = 1.0
    if totalPhotoelectrons > maxfill:
        scalefactor = maxfill / totalPhotoelectrons
    return scalefactor


def coldshieldSelfEmission(wavelengths, coldshieldTemperature, D, f):
    """For infrared systems, this term represents spectral irradiance on the
    FPA due to emissions from the walls of the dewar itself.

    :param wavelengths:
        wavelength array (m)
    :param coldshieldTemperature:
        temperature of the cold shield (K).  It is a common approximation to
        assume that the coldshield is at the same temperature as the detector
        array.
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        coldshieldE:
            cold shield spectral irradiance at the FPA (W / m^2 m)
    """
    # coldshield solid angle x blackbody emitted radiance
    coldshieldE = (
        np.pi - np.pi / (4.0 * (f / D) ** 2.0 + 1.0)
    ) * blackbodyRadiance(wavelengths, coldshieldTemperature)

    return coldshieldE


def coldstopSelfEmission(
    wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f
):
    """For infrared systems, this term represents spectral irradiance emitted
    by the cold stop on to the FPA.

    :param wavelengths:
        wavelength array (m)
    :param coldfilterTemperature:
        temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.
    :param coldfilterEmissivity:
        emissivity through the cold filter (unitless).  A common approximation
        is 1-cold filter transmission.
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        coldstopE:
            optics emitted irradiance on to the FPA (W / m^2 m)
    """

    coldstopL = coldfilterEmissivity * blackbodyRadiance(
        wavelengths, coldfilterTemperature
    )
    coldstopE = atFocalPlaneIrradiance(D, f, coldstopL)
    return coldstopE


def focalplaneIntegratedIrradiance(
    L, Ls, topt, eopt, lambda0, dlambda, opticsTemperature, D, f
):
    """IBSM Equation 3-34.  Calculates band integrated irradiance at the focal
    plane, including at-aperture scene radiance, optical self-emission, and
    non-thermal stray radiance.  NOTE: this function is only included for
    completeness. It is much better to use spectral quantities throughout the
    modeling process.

    :param L:
        band integrated at-aperture radiance (W/m^2 sr)
    :param Ls:
        band integrated stray radiance from sources other than self emission
        (W/m^2 sr)
    :param topt:
        full system in-band optical transmission (unitless).  If the telescope
        is obscured, topt is further reduced by 1-eta**2, where
        eta is the relative linear obscuration
    :param eopt:
        full system in-band optical emissivity (unitless).  1-topt is a good
        approximation.
    :param lambda0:
        wavelength at the center of the system bandpass (m)
    :param dlambda:
        system spectral bandwidth (m)
    :param opticsTemperature:
        temperature of the optics (K)
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        E:
            integrated irradiance (W/m^2) at the focal plane
    """
    L = (
        topt * L
        + eopt * blackbodyRadiance(lambda0, opticsTemperature) * dlambda
        + Ls
    )
    E = atFocalPlaneIrradiance(D, f, L)

    return E


def loadDatabaseAtmosphere_nointerp(altitude, groundRange, ihaze):
    """Loads a precalculated MODTRAN 5.2.1 Tape 7 atmosphere over a wavelength
    range of 0.3 to 14 micrometers.  All screnario details are in
    'atmosphere_README.txt'. NOTE: the _nointerp suffix was added for version
    0.2. See pybsm.loadDatabaseAtmosphere for more information.

    :param altitude:
        sensor height above ground level in meters.  The database includes the
        following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
        12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
        24500 meters
    :param groundRange:
        distance *on the ground* between the target and sensor in meters.
        The following ground ranges are included in the database at each
        altitude until the ground range exceeds the distance to the spherical
        earth horizon: 0 100 500 1000 to 20000 in 1000 meter steps, 22000 to
        80000 in 2000 m steps, and  85000 to 300000 in 5000 meter steps.
    :param ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural
        extinction with 23 km visibility) or ihaze = 2 (Rural extinction
        with 5 km visibility)


    :return:
        atm[:,0]:
            wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps
        atm[:,1]:
            (TRANS) total transmission through the defined path.
        atm[:,2]:
            (PTH THRML) radiance component due to atmospheric emission and
            scattering received at the observer.
        atm[:,3]:
            (SURF EMIS) component of radiance due to surface emission received
            at the observer.
        atm[:,4]:
            (SOL SCAT) component of scattered solar radiance received at the
            observer.
        atm[:,5]:
            (GRND RFLT) is the total solar flux impingent on the ground and
            reflected directly to the sensor from the ground. (direct radiance
            + diffuse radiance) * surface reflectance
    :NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)
    """

    # decoder maps filenames to atmospheric attributes
    atmpath = os.path.join(dirpath, "atms", "fileDecoder.csv")
    decoder = np.genfromtxt(atmpath, delimiter=",", skip_header=1)

    decoder = decoder[
        decoder[:, 3] == ihaze
    ]  # downselects to the right ihaze mode
    decoder = decoder[
        decoder[:, 1] == altitude / 1000.0
    ]  # downselects to the right altitude
    decoder = decoder[
        decoder[:, 2] == groundRange / 1000.0
    ]  # downselects to the right ground range

    rawdata = np.fromfile(
        dirpath + "/atms/" + str(int(decoder[0, 0])) + ".bin",
        dtype=np.float32,
        count=-1,
    )

    rawdata = rawdata.reshape((1371, 5), order="F")
    rawdata[:, 1:5] = (
        rawdata[:, 1:5] / 1e-10
    )  # convert radiance columns to W/(sr m^2 m)

    # append wavelength as first column
    wavl = 1e-6 * np.linspace(0.3, 14.0, 1371)
    wavl = np.expand_dims(wavl, axis=1)
    atm = np.hstack((wavl, rawdata))

    return atm


def loadDatabaseAtmosphere(altitude, groundRange, ihaze):
    """linear interpolation of the pre-calculated MODTRAN atmospheres.
    See the original 'loadDatabaseAtmosphere' (now commented out) for more
    details on the outputs.
    NOTE: This is experimental code.  Linear interpolation between atmospheres
    may not be a good approximation in every case!!!!


    :param altitude:
        sensor height above ground level in meters
    :param groundRange:
    :param ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural
        extinction with 23 km visibility) or ihaze = 2 (Rural extinction
        with 5 km visibility)

    :return:
        atm[:,0]:
            wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps
        atm[:,1]:
            (TRANS) total transmission through the defined path.
        atm[:,2]:
            (PTH THRML) radiance component due to atmospheric emission and
            scattering received at the observer.
        atm[:,3]:
            (SURF EMIS) component of radiance due to surface emission received
            at the observer.
        atm[:,4]:
            (SOL SCAT) component of scattered solar radiance received at the
            sobserver.
        atm[:,5]:
            (GRND RFLT) is the total solar flux impingent on the ground and
            reflected directly to the sensor from the ground. (direct radiance
            + diffuse radiance) * surface reflectance
    :NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)
    """

    def getGroundRangeArray(maxGroundRange):
        """Returns an array of ground ranges that are valid in the
        precalculated MODTRAN database.

        :param maxGroundRange:
            largest ground Range of interest (m)

        :return:
            G:
                array of ground ranges less than maxGroundRange (m)
        """
        G = np.array([0.0, 100.0, 500.0])
        G = np.append(G, np.arange(1000.0, 20000.01, 1000.0))
        G = np.append(G, np.arange(22000.0, 80000.01, 2000.0))
        G = np.append(G, np.arange(85000.0, 300000.01, 5000.0))
        G = G[G <= maxGroundRange]
        return G

    def altAtmInterp(lowalt, highalt, altitude, groundRange, ihaze):
        # this is an internal function for interpolating atmospheres across
        # altitudes
        lowatm = loadDatabaseAtmosphere_nointerp(lowalt, groundRange, ihaze)
        if lowalt != highalt:
            highatm = loadDatabaseAtmosphere_nointerp(
                highalt, groundRange, ihaze
            )
            lowweight = 1 - ((altitude - lowalt) / (highalt - lowalt))
            highweight = (altitude - lowalt) / (highalt - lowalt)
            atm = lowweight * lowatm + highweight * highatm
        else:
            atm = lowatm
        return atm

    # define arrays of all possible altitude and ground ranges
    altarray = np.array(
        [
            2,
            32.55,
            75,
            150,
            225,
            500,
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            11000,
            12000,
            14000,
            16000,
            18000,
            20000,
        ]
    )
    grangearray = getGroundRangeArray(301e3)

    # find the database altitudes and ground ranges that bound the values of
    # interest
    lowalt = altarray[altarray <= altitude][-1]
    highalt = altarray[altarray >= altitude][0]
    lowrng = grangearray[grangearray <= groundRange][-1]
    highrng = grangearray[grangearray >= groundRange][0]

    # first interpolate across the low and high altitudes
    # then interpolate across ground range
    atm_lowrng = altAtmInterp(lowalt, highalt, altitude, lowrng, ihaze)
    if lowrng != highrng:
        atm_highrng = altAtmInterp(lowalt, highalt, altitude, highrng, ihaze)
        lowweight = 1 - ((groundRange - lowrng) / (highrng - lowrng))
        highweight = (groundRange - lowrng) / (highrng - lowrng)
        atm = lowweight * atm_lowrng + highweight * atm_highrng
    else:
        atm = atm_lowrng

    return atm


def opticsSelfEmission(
    wavelengths,
    opticsTemperature,
    opticsEmissivity,
    coldfilterTransmission,
    D,
    f,
):
    """For infrared systems, this term represents spectral irradiance emitted
    by the optics (but not the cold stop) on to the FPA.

    :param wavelengths:
        wavelength array (m)
    :param opticsTemperature:
        temperature of the optics (K)
    :param opticsEmissivity:
        emissivity of the optics (unitless) except for the cold filter.
        A common approximation is 1-optics transmissivity.
    :param coldfilterTransmission:
        transmission through the cold filter (unitless)
    :param D:
        effective aperture diameter (m)
    :param f:
        focal length (m)

    :return:
        opticsE:
            optics emitted irradiance on to the FPA (W / m^2 m)
    """

    opticsL = (
        coldfilterTransmission
        * opticsEmissivity
        * blackbodyRadiance(wavelengths, opticsTemperature)
    )
    opticsE = atFocalPlaneIrradiance(D, f, opticsL)
    return opticsE


def photonDetectionRate(E, wx, wy, wavelengths, qe):
    """IBSM Equation 3-42 with dark current, integration time, and tdi
    separated out. Conversion of photons into photoelectrons.  There is a
    possible disconnect here in the documentation. Equation 3-42 appears to be
    a spectral quantity but the documentation calls for an integrated
    irradiance.  It is definitely used here as a spectral quantity.

    :param E:
        spectral irradiance (W/m^2 m) at the focal plane at each wavelength
    :param wx:
        detector size (width) in the x direction (m)
    :param wy:
        detector size (width) in the y direction (m)
    :param wavelengths:
        wavelength array (m)
    :param qe:
        quantum efficiency (e-/photon)

    :return:
        dN:
            array of photoelectrons/wavelength/second (e-/m)
    :NOTE:
        To calculate total integrated photoelectrons, N = td*ntdi*np.trapz(dN,
        wavelens) where td is integration time (s) and ntdi is the number of
        tdi stages (optional)
    """

    dN = (wavelengths / (hc * cc)) * qe * wx * wy * E
    return dN


def photonDetectorSNR(
    sensor, radianceWavelengths, targetRadiance, backgroundRadiance
):
    """Calculates extended target contrast SNR for semiconductor- based photon
    detector systems (as opposed to thermal detectors).  This code originally
    served the NIIRS model but has been abstracted for other uses.  Photon,
    dark current, quantization, and read noise are all explicitly considered.
    You can also pass in other noise terms (as rms photoelectrons) as a numpy
    array sensor.otherNoise.

    :param sensor:
        an object from the class sensor
    :param radianceWavelengths:
        a numpy array of wavelengths (m)
    :param targetRadiance:
        a numpy array of target radiance values corresponding to
        radianceWavelengths (W/m^2 sr m)
    :param backgroundRadiance:
        a numpy array of target radiance values corresponding to
        radianceWavelengths (W/m^2 sr m)

    :return:
        snr:
            an object containing results of the SNR calculation along with many
            intermediate calculations.  The SNR value is contained in snr.snr
    """
    snr = metrics.Metrics("signal-to-noise calculation")

    # resample the optical transmission and quantum efficiency functions
    snr.optTrans = (
        sensor.coldfilterTransmission
        * (1.0 - sensor.eta**2)
        * resampleByWavelength(
            sensor.optTransWavelengths,
            sensor.opticsTransmission,
            radianceWavelengths,
        )
    )
    snr.qe = resampleByWavelength(
        sensor.qewavelengths, sensor.qe, radianceWavelengths
    )

    # for infrared systems, calculate FPA irradiance contributions from within
    # the sensor system itself
    snr.otherIrradiance = (
        coldshieldSelfEmission(
            radianceWavelengths,
            sensor.coldshieldTemperature,
            sensor.D,
            sensor.f,
        )
        + opticsSelfEmission(
            radianceWavelengths,
            sensor.opticsTemperature,
            sensor.opticsEmissivity,
            sensor.coldfilterTransmission,
            sensor.D,
            sensor.f,
        )
        + coldstopSelfEmission(
            radianceWavelengths,
            sensor.coldfilterTemperature,
            sensor.coldfilterEmissivity,
            sensor.D,
            sensor.f,
        )
    )

    # initial estimate of  total detected target and background photoelectrons
    # first target.  Note that snr.weights is useful for later calculations
    # that require weighting as a function of wavelength (e.g. aperture OTF)
    snr.tgtNrate, snr.tgtFPAirradiance, snr.weights = signalRate(
        radianceWavelengths,
        targetRadiance,
        snr.optTrans,
        sensor.D,
        sensor.f,
        sensor.wx,
        sensor.wy,
        snr.qe,
        snr.otherIrradiance,
        sensor.darkCurrent,
    )
    snr.tgtN = snr.tgtNrate * sensor.intTime * sensor.ntdi
    # then background
    snr.bkgNrate, snr.bkgFPAirradiance, _ = signalRate(
        radianceWavelengths,
        backgroundRadiance,
        snr.optTrans,
        sensor.D,
        sensor.f,
        sensor.wx,
        sensor.wy,
        snr.qe,
        snr.otherIrradiance,
        sensor.darkCurrent,
    )
    snr.bkgN = snr.bkgNrate * sensor.intTime * sensor.ntdi

    # check to see that well fill is within a desirable range and, if not,
    # scale back the integration time and recalculate the total photon counts
    scalefactor = checkWellFill(
        np.max([snr.tgtN, snr.bkgN]), sensor.maxWellFill * sensor.maxN
    )
    snr.tgtN = scalefactor * snr.tgtN
    snr.bkgN = scalefactor * snr.bkgN
    snr.intTime = scalefactor * sensor.intTime
    snr.wellfraction = np.max([snr.tgtN, snr.bkgN]) / sensor.maxN
    # another option would be to reduce TDI stages if applicable, this should
    # be a concern if TDI mismatch MTF is an issue

    # calculate contrast signal (i.e. target difference above or below the
    # background)
    snr.contrastSignal = snr.tgtN - snr.bkgN

    # break out noise terms (rms photoelectrons)
    # signalNoise includes scene photon noise, dark current noise, and self
    # emission noise
    snr.signalNoise = np.sqrt(np.max([snr.tgtN, snr.bkgN]))
    # just noise from dark current
    snr.darkcurrentNoise = np.sqrt(
        sensor.ntdi * sensor.darkCurrent * snr.intTime
    )
    # quantization noise
    snr.quantizationNoise = noise.quantizationNoise(
        sensor.maxN, sensor.bitdepth
    )
    # photon noise due to self emission in the optical system
    snr.selfEmissionNoise = np.sqrt(
        np.trapz(
            photonDetectionRate(
                snr.otherIrradiance,
                sensor.wx,
                sensor.wy,
                radianceWavelengths,
                snr.qe,
            ),
            radianceWavelengths,
        )
        * snr.intTime
        * sensor.ntdi
    )

    # note that signalNoise includes sceneNoise, dark current noise, and self
    # emission noise
    snr.totalNoise = np.sqrt(
        snr.signalNoise**2
        + snr.quantizationNoise**2
        + sensor.readNoise**2
        + np.sum(sensor.otherNoise**2)
    )

    # calculate signal-to-noise ratio
    snr.snr = snr.contrastSignal / snr.totalNoise

    return snr


def reflectance2photoelectrons(atm, sensor, intTime, target_temp=300):
    """Provides a mapping between reflectance (0 to 1 in 100 steps) on the
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
        atmospheric data as defined in loadDatabaseAtmosphere.  The slant
        range between the target and sensor are implied by this choice.
    :param sensor:
        sensor parameters as defined in the pybsm sensor class
    :param intTime:
        camera integration time (s).
    :param target_temp: float
        Temperature of the target (Kelvin).

    :return:
        ref :
            array of reflectance values (unitless) from 0 to 1 in 100 steps
        pe :
            photoelectrons generated during the integration time corresponding
            to the reflectance values in ref
        spectral_weights : 2xN arraylike
            Details of the relative spectral contributions to the collected
            signal, which is useful for wavelength-weighted OTF calculations.
            The first row is the wavelength (m) and the second row is the
            relative contribution to the signal from the associated
            column-paired wavelength.
    """

    ref = np.linspace(0.0, 1.0, 100)
    pe = np.zeros(ref.shape)
    atm = atm[atm[:, 0] >= sensor.optTransWavelengths[0], :]
    atm = atm[atm[:, 0] <= sensor.optTransWavelengths[-1], :]

    for idx in np.arange(ref.size):
        # Calculate the total radiance from the target including both
        # reflection of the solar illumination and the radiative emission from
        # the object assumed to be at 300 K.
        targetRadiance = totalRadiance(atm, ref[idx], 300.0)

        wavelengths = atm[:, 0]

        optTrans = (
            sensor.coldfilterTransmission
            * (1.0 - sensor.eta**2)
            * resampleByWavelength(
                sensor.optTransWavelengths,
                sensor.opticsTransmission,
                wavelengths,
            )
        )

        qe = resampleByWavelength(sensor.qewavelengths, sensor.qe, wavelengths)

        # The components of the imaging system is at a non-zero temperature and
        # itself generates radiative emissions. So, we account for these
        # emissions here. This is only relevant in the thermal infrared bands.
        otherIrradiance = coldshieldSelfEmission(
            wavelengths, sensor.coldshieldTemperature, sensor.D, sensor.f
        )
        otherIrradiance = otherIrradiance + opticsSelfEmission(
            wavelengths,
            sensor.opticsTemperature,
            sensor.opticsEmissivity,
            sensor.coldfilterTransmission,
            sensor.D,
            sensor.f,
        )
        otherIrradiance = otherIrradiance + coldstopSelfEmission(
            wavelengths,
            sensor.coldfilterTemperature,
            sensor.coldfilterEmissivity,
            sensor.D,
            sensor.f,
        )

        tgtNrate, tgtFPAirradiance, weights = signalRate(
            wavelengths,
            targetRadiance,
            optTrans,
            sensor.D,
            sensor.f,
            sensor.wx,
            sensor.wy,
            qe,
            otherIrradiance,
            sensor.darkCurrent,
        )

        pe[idx] = tgtNrate * intTime * sensor.ntdi

    sat = pe.max() / sensor.maxN
    if sat > 1:
        print(
            f"Reducing integration time from {intTime} to {intTime/sat}"
            "to avoid overexposure"
        )
        pe = pe / sat

    # Clip to the maximum number of photoelectrons that can be held.
    pe[pe > sensor.maxN] = sensor.maxN

    spectral_weights = np.vstack([wavelengths, weights / max(weights)])

    return ref, pe, spectral_weights


def signalRate(
    wavelengths,
    targetRadiance,
    opticalTransmission,
    D,
    f,
    wx,
    wy,
    qe,
    otherIrradiance,
    darkCurrent,
):
    """For semiconductor-based detectors, returns the signal rate (total
    photoelectrons/s) generated at the output of the detector along with a
    number of other related quantities.  Multiply this quantity by the
    integration time (and the number of TDI stages, if applicable) to determine
    the total number of detected photoelectrons.

    :param wavelengths:
        array of wavelengths (m)
    :param targetRadiance:
        apparent target spectral radiance at the aperture including all
        atmospheric contributions (W/sr m^2 m)
    :param backgroundRadiance:
        apparent background spectral radiance at the aperture including all
        atmospheric contributions (W/sr m^2 m)
    :param opticalTransmission:
        transmission of the telescope optics as a function of wavelength
        (unitless)
    :param D:
        effective aperture diameter (m)
    :param (wx,wy):
        detector size (width) in the x and y directions (m)
    :param f:
        focal length (m)
    :param qe:
        quantum efficiency as a function of wavelength (e-/photon)
    :param otherIrradiance:
        spectral irradiance from other sources (W/m^2 m).
        This is particularly useful for self emission in infrared cameras.
        It may also represent stray light.
    :param darkCurrent:
        detector dark current (e-/s)

    :return:
        tgtRate:
            total integrated photoelectrons per seconds (e-/s)
        tgtFPAirradiance:
            spectral irradiance at the FPA (W/m^2 m)
        tgtdN:
            spectral photoelectrons (e-/s m)

    """
    # get at FPA spectral irradiance
    tgtFPAirradiance = (
        opticalTransmission * atFocalPlaneIrradiance(D, f, targetRadiance)
        + otherIrradiance
    )

    # convert spectral irradiance to spectral photoelectron rate
    tgtdN = photonDetectionRate(tgtFPAirradiance, wx, wy, wavelengths, qe)

    # calculate total detected target and background photoelectron rate
    tgtRate = np.trapz(tgtdN, wavelengths) + darkCurrent

    return tgtRate, tgtFPAirradiance, tgtdN


def totalRadiance(atm, reflectance, temperature):
    """Calculates total spectral radiance at the aperture for a object of
    interest.



    :param atm:
        matrix of atmospheric data (see loadDatabaseAtmosphere for details)
    :param reflectance:
        object reflectance (unitless)
    :param temperature:
        object temperature (Kelvin)

    :return:
        radiance:
            radiance = path thermal + surface emission + solar scattering +
            ground reflected (W/m^2 sr m)

    :NOTE:
        In the emissive infrared region (e.g. >= 3 micrometers), the nighttime
        case is very well approximated by subtracting off atm[:,4] from the
        total spectral radiance
    """

    dbreflectance = 0.15  # object reflectance used in the database
    radiance = (
        atm[:, 2]
        + (1.0 - reflectance)
        * blackbodyRadiance(atm[:, 0], temperature)
        * atm[:, 1]
        + atm[:, 4]
        + atm[:, 5] * (reflectance / dbreflectance)
    )

    return radiance


def giqeRadiance(atm, isEmissive):
    """This function provides target and background spectral radiance as
    defined by the GIQE.

    :param atm:
        an array containing the following data:
        atm[:,0] - wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps

        atm[:,1] - (TRANS) total transmission through the defined path.

        atm[:,2] - (PTH THRML) radiance component due to atmospheric emission
        and scattering received at the observer.

        atm[:,3] - (SURF EMIS) component of radiance due to surface emission
        received at the observer.

        atm[:,4] - (SOL SCAT) component of scattered solar radiance received
        at the observer.

        atm[:,5] - (GRND RFLT) is the total solar flux impingent on the
        ground and reflected directly to the sensor from the ground. (direct
        radiance + diffuse radiance) * surface reflectance

        NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)
    :param isEmissive:
        isEmissive = 1 for thermal emissive band NIIRS, otherwise
        isEmissive = 0

    :return:
        targetRadiance:
            apparent target spectral radiance at the aperture including all
            atmospheric contributions
        backgroundRadiance:
            apparent background spectral radiance at the aperture including
            all atmospheric contributions

    :NOTE:
        The nighttime emissive case is well approximated by subtracting off
        atm[:,4] from the returned values.

    """
    tgtTemp = 282.0  # target temperature (original GIQE suggestion was 282 K)
    bkgTemp = 280.0  # background temperature (original GIQE 280 K)
    tgtRef = 0.15  # percent reflectance of the target (should be .15 for GIQE)
    bkgRef = (
        0.07  # percent reflectance of the background (should be .07 for GIQE)
    )

    if isEmissive:
        # target and background are blackbodies
        targetRadiance = totalRadiance(atm, 0.0, tgtTemp)
        backgroundRadiance = totalRadiance(atm, 0.0, bkgTemp)
    else:
        targetRadiance = totalRadiance(atm, tgtRef, tgtTemp)
        backgroundRadiance = totalRadiance(atm, bkgRef, bkgTemp)

    return targetRadiance, backgroundRadiance


def resampleByWavelength(wavelengths, values, newWavelengths):
    """Resamples arrays that are input as a function of wavelength.

    :param wavelengths:
        array of wavelengths (m)
    :param values:
        array of values to be resampled (arb)
    :param newWavelengths:
        the desired wavelength range and step size (m)

    :return:
        newValues: array of values resampled to match newWavelengths.
        Extrapolated values are set to 0.
    """
    newValues = np.interp(newWavelengths, wavelengths, values, 0.0, 0.0)
    return newValues
