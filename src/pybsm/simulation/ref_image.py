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
# 3rd party imports
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .scenario import Scenario

# local imports
from .sensor import Sensor


class RefImage:
    """Reference image.

    :param img: Reference image.
    :param gsd:
        Spatial sampling for 'img' in meters. Each pixel in 'img' is assumed to
        capture a 'gsd' x 'gsd' square of some world surface. We assume the
        sampling is isotropic (x and y sampling are identical) and uniform
        across the whole field of view. This is generally a valid assumption
        for remote sensing imagery.
    :param pix_values:
        Pixel count values within 'img' that should be associated with the
        corresponding reflectance values in 'refl_values' by linear
        interpolation. This is used to convert raw image values into an assumed
        spectral reflectance of the scene being viewed.
    :param refl_values:
        Reflectance values associated with the corresponding pixel count values
        in 'pix_values' used to convert raw image values into an assumed
        spectral reflectance of the scene being viewed.
    :param name:    Name of the image.

    :raises: ValueError if pix_values is provided, but refl_values is missing
    """

    def __init__(
        self,
        img: np.ndarray,
        gsd: float,
        pix_values: Optional[np.ndarray] = None,
        refl_values: Optional[np.ndarray] = None,
        name: str = "ref_image",
    ) -> None:
        self.img = img
        self.gsd = gsd
        self.name = name

        if pix_values is None:
            pix_values = np.array(
                [
                    np.percentile(img.ravel(), 0.2),
                    np.percentile(img.ravel(), 99.8),
                ]
            )
            refl_values = np.array([0.05, 0.95])
        else:
            if refl_values is None:
                raise ValueError(
                    "If 'pix_values' is provided, 'refl_values' must be as well."
                )

        self.pix_values = pix_values
        self.refl_values = refl_values

    def estimate_capture_parameters(
        self, altitude: float = 2000000
    ) -> Tuple[Sensor, Scenario]:
        """Estimate the scenario and sensor parameters that are consistent with this image.

        This provides a no-degradation baseline from which to alter parameters
        to explore further degradation.
        """
        # Let's assume the full visible spectrum.
        opt_trans_wavelengths = np.array([380, 700]) * 1.0e-9  # m

        scenario = Scenario(
            self.name,
            1,
            altitude,
            ground_range=0,
            aircraft_speed=0,
            ha_wind_speed=0,
            cn2_at_1m=0,
        )

        # Guess at a relatively large pixel pitch, which should have a large
        # well depth.
        p = 20e-6  # m

        # Calculate the focal length (m) that matches the GSD for the
        # prescribed altitude and pixel pitch.
        f = altitude * p / self.gsd  # m

        # Instantenous field of view (iFOV), the angular extent of the world
        # covered by one pixel (radians).
        ifov = 2 * np.arctan(p / 2 / f)

        # We are assuming a circular aperture without obscuration. The
        # diffraction limited angular resolution (where on Airy disk sits in
        # the first ring of another Airy disk) is 1.22*lambda/D. But, let's use
        # a coefficient of 4 for safety.
        D = 4 * np.median(opt_trans_wavelengths) / ifov  # noqa: N806

        sensor = Sensor(self.name, D, f, p, opt_trans_wavelengths)
        return sensor, scenario

    def show(self) -> None:
        h, w = self.img.shape[:2]
        plt.imshow(
            self.img,
            extent=[
                -w / 2 * self.gsd,
                w / 2 * self.gsd,
                -h / 2 * self.gsd,
                h / 2 * self.gsd,
            ],
        )
        plt.xlabel("X-Position (m)", fontsize=24)
        plt.ylabel("Y-Position (m)", fontsize=24)
        plt.tight_layout()
