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
# 3rd party imports
import numpy as np
from typing import Optional

# local imports
from pybsm import utils


class Scenario:
    """Everything about the target and environment.

    NOTE:  if the niirs model
    is called, values for target/background temperature, reflectance, etc. are
    overridden with the NIIRS model defaults.

    :parameter ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural
        extinction with 23 km visibility) or ihaze = 2 (Rural extinction
        with 5 km visibility)
    :parameter altitude:
        sensor height above ground level in meters; the database includes the
        following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
        12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
        24500
    :parameter ground_range:
        projection of line of sight between the camera and target along on the
        ground in meters; the distance between the target and the camera is
        given by sqrt(altitude^2 + ground_range^2).
        The following ground ranges are included in the database at each
        altitude until the ground range exceeds the distance to the spherical
        earth horizon: 0 100 500 1000 to 20000 in 1000 meter steps, 22000 to
        80000 in 2000 m steps, and  85000 to 300000 in 5000 meter steps.
    :parameter aircraftSpeed:
        ground speed of the aircraft (m/s)
    :parameter targetReflectance:
        object reflectance (unitless); the default 0.15 is the giqe standard
    :parameter targetTemperature:
        object temperature (Kelvin); 282 K is used for GIQE calculation
    :parameter backgroundReflectance:
        background reflectance (unitless)
    :parameter backgroundTemperature:
        background temperature (Kelvin); 280 K used for GIQE calculation
    :parameter haWindspeed:
        the high altitude windspeed (m/s) used to calculate the turbulence
        profile; the default, 21.0, is the HV 5/7 profile value
    :parameter cn2at1m:
        the refractive index structure parameter "near the ground"
        (e.g. at h = 1 m) used to calculate the turbulence profile; the
        default, 1.7e-14, is the HV 5/7 profile value

    """

    def __init__(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        aircraftSpeed: float = 0,
        targetReflectance: float = 0.15,
        targetTemperature: float = 295,
        backgroundReflectance: float = 0.07,
        backgroundTemperature: float = 293,
        haWindspeed: float = 21,
        cn2at1m: float = 1.7e-14
    ) -> None:
        self.name = name
        self._ihaze = ihaze
        self._altitude = altitude
        self._ground_range = ground_range
        self.aircraftSpeed = aircraftSpeed
        self.targetReflectance = targetReflectance
        self.targetTemperature = targetTemperature
        self.backgroundReflectance = backgroundReflectance
        self.backgroundTemperature = backgroundTemperature
        self.haWindspeed = haWindspeed
        self.cn2at1m = cn2at1m

        # Will be loaded on demand for a particular altitude.
        self._atm: Optional[np.ndarray] = None

    @property
    def ihaze(self) -> int:
        return self._ihaze

    @ihaze.setter
    def ihaze(self, value: int) -> None:
        self._atm = None
        self._ihaze = value

    @property
    def altitude(self) -> float:
        return self._altitude

    @altitude.setter
    def altitude(self, value: float) -> None:
        self._atm = None
        self._altitude = value

    @property
    def ground_range(self) -> float:
        return self._ground_range

    @ground_range.setter
    def ground_range(self, value: float) -> None:
        self._atm = None
        self._ground_range = value

    @property
    def atm(self) -> np.ndarray:
        """Return atmospheric spectral absorption.

        :return:
            List of values:

            atm[:,0]- wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps

            atm[:,1]- (TRANS) total transmission through the defined path

            atm[:,2]- (PTH THRML) radiance component due to atmospheric
            emission and scattering received at the observer

            atm[:,3]- (SURF EMIS) component of radiance due to surface
            emission received at the observer

            atm[:,4]- (SOL SCAT) component of scattered solar radiance
            received at the observer

            atm[:,5]- (GRND RFLT) is the total solar flux impingement on the
            ground and reflected directly to the sensor from the ground.
            (direct radiance + diffuse radiance) * surface reflectance

            NOTE: Units for columns 1 through 5 are in radiance W/(sr m^2 m).

        """
        if self._atm is None:
            # Read in and cache results.
            self._atm = utils.loadDatabaseAtmosphere(
                self.altitude, self.ground_range, self.ihaze
            )

        return self._atm
