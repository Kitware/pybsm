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


def loadDatabaseAtmosphere_nointerp(
    altitude: float,
    groundRange: float,
    ihaze: int
) -> np.ndarray:
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


def loadDatabaseAtmosphere(
    altitude: float,
    groundRange: float,
    ihaze: int
) -> np.ndarray:
    """linear interpolation of the pre-calculated MODTRAN atmospheres.
    See the original 'loadDatabaseAtmosphere' (now commented out) for more
    details on the outputs.
    NOTE: This is experimental code.  Linear interpolation between atmospheres
    may not be a good approximation in every case!!!!


    :param altitude:
        sensor height above ground level in meters
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
            sobserver.
        atm[:,5]:
            (GRND RFLT) is the total solar flux impingent on the ground and
            reflected directly to the sensor from the ground. (direct radiance
            + diffuse radiance) * surface reflectance
    :NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)
    """

    def getGroundRangeArray(maxGroundRange: float) -> np.ndarray:
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

    def altAtmInterp(
        lowalt: float,
        highalt: float,
        altitude: float,
        groundRange: float,
        ihaze: int
    ) -> np.ndarray:
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
