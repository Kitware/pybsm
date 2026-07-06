from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image
from scipy import interpolate
from syrupy.assertion import SnapshotAssertion

from pybsm import simulation
from pybsm.simulation import (
    CircularApertureSimulator,
    DefocusSimulator,
    DetectorSimulator,
    JitterSimulator,
    SystemOTFSimulator,
    TurbulenceApertureSimulator,
)

BASE_FILE_PATH = Path(__file__).parent.parent.parent
IMAGE_FILE_PATH = (
    BASE_FILE_PATH / "docs" / "examples" / "data" / "M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)


class TestImageSimulator:
    @pytest.mark.parametrize(
        ("img_file_path", "use_reflectance", "reflectance_range", "mtf_wavelengths", "mtf_weights"),
        [
            (  # use_reflectance is True, but no reflectance_range is provided
                IMAGE_FILE_PATH,
                True,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # reflectance_range length is not 2
                IMAGE_FILE_PATH,
                True,
                np.array([1]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # reflectance_range is not strictly increasing
                IMAGE_FILE_PATH,
                True,
                np.array([0.5, 0.05]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # mtf_wavelengths and mtf_weights are not equal length
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5]),
            ),
            (  # mtf_wavelengths is empty
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([]),
                np.array([0.5, 0.5]),
            ),
            (  # mtf_weights is empty
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([]),
            ),
        ],
    )
    def test_init_value_error(
        self,
        img_file_path: str,
        use_reflectance: bool,
        reflectance_range: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
    ) -> None:
        """Cover cases where ValueError occurs."""
        img = np.array(Image.open(img_file_path))
        gsd = 3.19 / 160.0
        altitude = 1000
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)
        with pytest.raises(ValueError):  # noqa: PT011
            SystemOTFSimulator(
                sensor=sensor,
                scenario=scenario,
                use_reflectance=use_reflectance,
                reflectance_range=reflectance_range,
                mtf_wavelengths=mtf_wavelengths,
                mtf_weights=mtf_weights,
            )


class TestSystemOTFSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))

    def test_psf_caching(self) -> None:
        """Verify PSF caching behavior."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        assert len(simulator._psf_cache) == 0

        psf1 = simulator._get_psf_cached(gsd=gsd)
        assert len(simulator._psf_cache) == 1

        psf2 = simulator._get_psf_cached(gsd=gsd)
        assert len(simulator._psf_cache) == 1
        assert np.array_equal(psf1, psf2)

        psf3 = simulator._get_psf_cached(gsd=gsd * 2)
        assert len(simulator._psf_cache) == 2
        assert not np.array_equal(psf1, psf3)

    def test_slant_range_altitude_overrides(self) -> None:
        """Verify slant_range and altitude override parameters."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        custom_slant_range = 1500.0
        custom_altitude = 1200.0

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
            slant_range=custom_slant_range,
            altitude=custom_altitude,
        )

        assert simulator.slant_range == custom_slant_range
        assert simulator._altitude == custom_altitude

    def test_sensor_scenario_configuration(self) -> None:
        """Verify sensor and scenario attributes match input configurations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        assert simulator.sensor.name == sensor.name
        assert simulator.sensor.D == sensor.D
        assert simulator.sensor.f == sensor.f
        assert simulator.sensor.p_x == sensor.p_x
        assert np.array_equal(simulator.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths)
        assert np.array_equal(simulator.sensor.optics_transmission, sensor.optics_transmission)
        assert simulator.sensor.eta == sensor.eta
        assert simulator.sensor.w_x == sensor.w_x
        assert simulator.sensor.w_y == sensor.w_y
        assert simulator.sensor.int_time == sensor.int_time
        assert simulator.sensor.dark_current == sensor.dark_current
        assert simulator.sensor.read_noise == sensor.read_noise
        assert simulator.sensor.max_n == sensor.max_n
        assert simulator.sensor.bit_depth == sensor.bit_depth
        assert simulator.sensor.max_well_fill == sensor.max_well_fill
        assert simulator.sensor.s_x == sensor.s_x
        assert simulator.sensor.s_y == sensor.s_y
        assert simulator.sensor.da_x == sensor.da_x
        assert simulator.sensor.da_y == sensor.da_y
        assert np.array_equal(simulator.sensor.qe_wavelengths, sensor.qe_wavelengths)
        assert np.array_equal(simulator.sensor.qe, sensor.qe)

        assert simulator.scenario.name == scenario.name
        assert simulator.scenario.ihaze == scenario.ihaze
        assert simulator.scenario.altitude == scenario.altitude
        assert simulator.scenario.ground_range == scenario.ground_range
        assert simulator.scenario.aircraft_speed == scenario.aircraft_speed
        assert simulator.scenario.target_reflectance == scenario.target_reflectance
        assert simulator.scenario.target_temperature == scenario.target_temperature
        assert simulator.scenario.background_reflectance == scenario.background_reflectance
        assert simulator.scenario.background_temperature == scenario.background_temperature
        assert simulator.scenario.ha_wind_speed == scenario.ha_wind_speed
        assert simulator.scenario.cn2_at_1m == scenario.cn2_at_1m

    def test_noise_reproducibility_with_seed(self) -> None:
        """Verify seed reproducibility."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert np.array_equal(noisy_img1, noisy_img2)

    def test_noise_reproducibility_with_generator(self) -> None:
        """Verify Generator seed reproducibility."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=np.random.default_rng(seed),
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=np.random.default_rng(seed),
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert np.array_equal(noisy_img1, noisy_img2)

    def test_noise_different_seeds(self) -> None:
        """Verify different seeds produce different outputs."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed + 1,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert not np.array_equal(noisy_img1, noisy_img2)


class TestJitterSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = JitterSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestCircularApertureSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = CircularApertureSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestDetectorSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = DetectorSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestDefocusSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = DefocusSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestTurbulenceApertureSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = TurbulenceApertureSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


def _build_simulator(
    *,
    use_reflectance: bool = True,
    reflectance_range: np.ndarray | None = None,
    altitude: int = 1000,
    sensor_overrides: dict | None = None,
) -> tuple[SystemOTFSimulator, np.ndarray]:
    """Construct a simulator + image pair matching the existing test convention."""
    img = np.array(Image.open(IMAGE_FILE_PATH))
    gsd = 3.19 / 160.0
    ref_img = simulation.RefImage(img=img, gsd=gsd)
    sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)
    if sensor_overrides:
        for key, value in sensor_overrides.items():
            setattr(sensor, key, value)
    if reflectance_range is None and use_reflectance:
        reflectance_range = ref_img.refl_values
    simulator = SystemOTFSimulator(
        sensor=sensor,
        scenario=scenario,
        use_reflectance=use_reflectance,
        reflectance_range=reflectance_range,
    )
    return simulator, img


def _pe_with_5pct_below_floor(simulator: SystemOTFSimulator, total: int = 1000) -> np.ndarray:
    """Construct a pe array with the first 5% of elements strictly below the forward floor."""
    pe = np.full(total, (simulator._fwd_y_min + simulator._fwd_y_max) / 2.0)
    pe[: int(0.05 * total)] = simulator._fwd_y_min - 1.0
    return pe


def _build_saturated_simulator() -> tuple[SystemOTFSimulator, np.ndarray]:
    """Build a simulator whose forward grid has a flattened (saturated) tail.

    Used by tests that exercise the non-monotonic-grid fallback path —
    the analytical inverse cannot be trusted when the forward grid is
    not strictly monotonic.
    """
    simulator, img = _build_simulator(
        sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
    )
    ref_grid = np.linspace(0.0, 1.0, 100)
    pe_grid = np.linspace(0.0, 30000.0, 100)
    pe_grid[80:] = pe_grid[80]
    simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_grid)
    simulator._init_inverse_coefs(ref_grid, pe_grid)
    return simulator, img


def _setup_clip_warning_case() -> tuple[SystemOTFSimulator, np.ndarray]:
    """Return a simulator + pe array that will trigger the clip-fraction warning."""
    simulator, _img = _build_simulator()
    return simulator, _pe_with_5pct_below_floor(simulator)


def _setup_fallback_warning_case() -> tuple[SystemOTFSimulator, np.ndarray]:
    """Return a simulator + pe array that will trigger the non-monotonic fallback warning."""
    simulator, _img = _build_saturated_simulator()
    return simulator, np.linspace(0.0, 30000.0, 64)


class TestPhotoelectronsToPixels:
    """Tests for ``ImageSimulator``'s photoelectron inverse pipeline."""

    def test_round_trip_machine_epsilon(self) -> None:
        """Round-tripping pe through the inverse and back must not lose precision.

        Drift in the analytical inverse would cause silent divergence
        in downstream code that compares photoelectron and reflectance
        values — a subtle precision regression that's hard to spot
        without this round-trip check.
        """
        simulator, _img = _build_simulator()
        fwd = simulator._reflect_to_photoelectrons
        pe_grid = np.linspace(simulator._fwd_y_min, simulator._fwd_y_max, 1000)
        ref_recovered = simulator.photoelectrons_to_reflectance(pe_grid)
        pe_round_trip = fwd(ref_recovered)
        np.testing.assert_allclose(pe_round_trip, pe_grid, rtol=1e-13, atol=0.0)

    def test_analytical_matches_interp1d_argument_swap(self) -> None:
        """The endpoint-fit inverse must agree with a direct interpolated inverse.

        The closed-form inverse formula and a direct
        ``interp1d(fwd.y, fwd.x)`` must produce equivalent outputs;
        divergence between them would mean the formula has silently
        drifted from the canonical interpolated reference.
        """
        simulator, _img = _build_simulator()
        fwd = cast(interpolate.interp1d, simulator._reflect_to_photoelectrons)
        ref_inv = interpolate.interp1d(fwd.y, fwd.x)
        pe_sample = np.linspace(simulator._fwd_y_min, simulator._fwd_y_max, 100_000)
        analytical = simulator.photoelectrons_to_reflectance(pe_sample, clip_to_unit=False)
        reference = ref_inv(pe_sample)
        np.testing.assert_allclose(analytical, reference, rtol=0.0, atol=1e-12)

    def test_clip_to_unit_false_allows_out_of_range_reflectance(self) -> None:
        """Opting out of the [0, 1] clip must let the inverse return values outside [0, 1].

        Callers that want to inspect how far their input fell outside the
        physical reflectance range need the unclipped result; the default
        clip would collapse "well below 0" and "exactly 0" into the same
        output, hiding that information.
        """
        simulator, _img = _build_simulator()
        affine_a, affine_b = simulator._affine_pe_coefs
        below = simulator.photoelectrons_to_reflectance(
            np.array([affine_b - abs(affine_a) * 0.5]),
            clip_to_unit=False,
        )
        above = simulator.photoelectrons_to_reflectance(
            np.array([affine_b + affine_a * 1.5]),
            clip_to_unit=False,
        )
        assert below[0] < 0.0
        assert above[0] > 1.0
        # Default clip pins both to [0, 1].
        clipped_below = simulator.photoelectrons_to_reflectance(
            np.array([affine_b - abs(affine_a) * 0.5]),
        )
        clipped_above = simulator.photoelectrons_to_reflectance(
            np.array([affine_b + affine_a * 1.5]),
        )
        assert clipped_below[0] == 0.0
        assert clipped_above[0] == 1.0

    def test_use_reflectance_false_raises_for_pe_to_reflectance(self) -> None:
        """Reflectance inverse must refuse when no reflectance forward map exists.

        Without ``use_reflectance=True`` there is no forward map to invert;
        the caller should be redirected to the ADC path on
        ``photoelectrons_to_pixels`` rather than receiving silently-wrong
        output.
        """
        simulator, _img = _build_simulator(use_reflectance=False)
        with pytest.raises(ValueError, match="use_reflectance=True"):
            simulator.photoelectrons_to_reflectance(np.array([1000.0]))

    def test_use_reflectance_false_routes_to_adc_in_pe_to_pixels(self) -> None:
        """Raw-pixel simulators must go through the ADC quantizer, not the inverse formula.

        With ``use_reflectance=False`` there is no forward map for the
        inverse formula to invert. Any dispatch that did not route to the
        ADC path would either crash or produce silently-wrong output.
        """
        simulator, _img = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        well_cap = 32400 * 1.0
        pe_sweep = np.linspace(0.0, well_cap, 64)
        pixels = simulator.photoelectrons_to_pixels(pe_sweep)
        assert np.all(np.diff(pixels) >= 0.0)
        assert pixels[0] == pytest.approx(0.0, abs=1.0)
        assert pixels[-1] == pytest.approx(255.0, abs=1.0)

    def test_adc_bit_depth_4_posterizes_to_at_most_16_levels(self) -> None:
        """ADC bit-depth must actually quantize the output, not just remap it.

        A 4-bit ADC can physically produce at most 16 unique output
        levels. A bypassed quantization stage (e.g. a rounding bug) would
        let the output silently take on up to 256 levels and make the
        ``bit_depth`` setting a no-op.
        """
        simulator, _img = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 4.0, "max_well_fill": 1.0},
        )
        pe_sweep = np.linspace(0.0, 32400.0, 1024)
        pixels = simulator.photoelectrons_to_pixels(pe_sweep)
        unique_uint8 = np.unique(np.clip(pixels, 0, 255).astype(np.uint8))
        assert unique_uint8.size <= 16

    def test_adc_bit_depth_non_integer_truncates_consistently(self) -> None:
        """Fractional bit-depth must behave identically to its truncated integer value.

        Real ADCs only have integer bit counts; pyBSM's default of 100.0
        is a placeholder. Treating ``bit_depth`` as a continuous knob
        (e.g. via ``ceil`` or interpolation) would make 12.5 and 12
        produce different output, breaking the physical contract.
        """
        sim_12, _ = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        sim_125, _ = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.5, "max_well_fill": 1.0},
        )
        pe_sweep = np.linspace(0.0, 32400.0, 256)
        np.testing.assert_array_equal(
            sim_12.photoelectrons_to_pixels(pe_sweep),
            sim_125.photoelectrons_to_pixels(pe_sweep),
        )

    @pytest.mark.parametrize(
        ("sensor_overrides", "match_pattern"),
        [
            ({"max_n": 0, "max_well_fill": 1.0}, "degenerate sensor well capacity"),
            ({"max_n": 32400, "bit_depth": 0.0, "max_well_fill": 1.0}, "bit_depth must be >= 1"),
        ],
        ids=["zero_well_capacity", "bit_depth_below_one"],
    )
    def test_adc_degenerate_sensor_raises(self, sensor_overrides: dict, match_pattern: str) -> None:
        """Impossible ADC configurations must raise, not return divide-by-zero noise.

        Zero well capacity or sub-1 bit-depth would otherwise propagate as
        NaN/Inf into the pixel output, silently corrupting downstream code.
        """
        simulator, _img = _build_simulator(use_reflectance=False, sensor_overrides=sensor_overrides)
        with pytest.raises(ValueError, match=match_pattern):
            simulator.photoelectrons_to_pixels(np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        ("p1", "p2"),
        [(0.0, 0.0), (10.0, 5.0)],
        ids=["equal", "inverted"],
    )
    def test_pe_to_pixels_invalid_output_range_raises(self, p1: float, p2: float) -> None:
        """Equal or inverted output ranges must raise, not produce degenerate output.

        ``p1 == p2`` would divide by zero in the remap; ``p1 > p2`` would
        produce a backwards mapping. Both fail fast with a clear message.
        """
        simulator, _img = _build_simulator()
        with pytest.raises(ValueError, match="strictly ascending"):
            simulator.photoelectrons_to_pixels(np.array([1000.0, 2000.0]), p1=p1, p2=p2)

    def test_raises_on_non_finite_input(self) -> None:
        """Non-finite pe must raise rather than silently casting to 0.

        A silent ``NaN -> 0`` cast on the final ``uint8`` step would corrupt
        the output with no visible signal to the caller.
        """
        simulator, _img = _build_simulator()
        pe_img = np.full(64, simulator._fwd_y_min, dtype=np.float64)
        pe_img[0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            simulator.photoelectrons_to_pixels(pe_img)

    @pytest.mark.parametrize(
        ("pe_factory", "expected_frac_lo", "expected_frac_hi", "expects_warning"),
        [
            (lambda sim: _pe_with_5pct_below_floor(sim), 0.05, 0.0, True),
            (lambda sim: np.linspace(sim._fwd_y_min, sim._fwd_y_max, 200), 0.0, 0.0, False),
            (lambda sim: np.full(100, sim._fwd_y_min), 0.0, 0.0, False),
        ],
        ids=[
            "5pct_below_floor_warns",
            "all_inside_silent",
            "exactly_at_floor_silent",
        ],
    )
    def test_clip_fraction_observability(
        self,
        pe_factory: Callable[[SystemOTFSimulator], np.ndarray],
        expected_frac_lo: float,
        expected_frac_hi: float,
        expects_warning: bool,
    ) -> None:
        """The clip-fraction counter and 1% warning must reflect actual boundary engagement.

        If the counter were inaccurate, callers would have no signal that
        their input was being clipped to the simulator's forward range
        (which silently saturates the analytical inverse).
        """
        simulator, _img = _build_simulator()
        pe_img = pe_factory(simulator)
        if expects_warning:
            with pytest.warns(UserWarning, match="clipped to forward-table bounds"):
                simulator.photoelectrons_to_pixels(pe_img)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                simulator.photoelectrons_to_pixels(pe_img)
        frac_lo, frac_hi = simulator._last_clip_fraction
        assert frac_lo == pytest.approx(expected_frac_lo, abs=1e-6)
        assert frac_hi == pytest.approx(expected_frac_hi, abs=1e-6)

    def test_warns_and_falls_back_on_nonmonotonic_forward_grid(self) -> None:
        """A saturated forward map must trigger the documented ADC fallback.

        The inverse formula is unreliable on a non-monotonic grid; the
        method's contract is to fall back to ADC quantization and warn
        once, rather than silently return wrong output.
        """
        simulator, _img = _build_saturated_simulator()
        pe_img = np.linspace(0.0, 30000.0, 64)
        with pytest.warns(UserWarning, match="forward_nonmonotonic"):
            pixels = simulator.photoelectrons_to_pixels(pe_img)
        expected = simulator._adc_photoelectrons_to_pixels(pe_img, out_range=(0.0, 255.0))
        np.testing.assert_array_equal(pixels, expected)

    @pytest.mark.parametrize(
        ("setup_fn", "warning_match"),
        [
            (_setup_clip_warning_case, "clipped to forward-table bounds"),
            (_setup_fallback_warning_case, "forward_nonmonotonic"),
        ],
        ids=["clip_warning", "fallback_warning"],
    )
    def test_warning_is_sticky_after_first_emission(
        self,
        setup_fn: Callable[[], tuple[SystemOTFSimulator, np.ndarray]],
        warning_match: str,
    ) -> None:
        """Each observability warning must fire at most once per simulator instance.

        Re-emitting on every call would flood downstream logs with the same
        message; the user just needs to know the condition was detected once.
        """
        simulator, pe_img = setup_fn()
        with pytest.warns(UserWarning, match=warning_match):
            simulator.photoelectrons_to_pixels(pe_img)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            simulator.photoelectrons_to_pixels(pe_img)

    def test_reflectance_warns_on_nonmonotonic_forward_grid(self) -> None:
        """Direct calls to the reflectance inverse must also flag a saturated forward map.

        The reflectance entry point has no fallback path; without this
        warning, a caller using a saturated forward map would silently
        get a degraded result.
        """
        simulator, _img = _build_saturated_simulator()
        with pytest.warns(UserWarning, match="forward map is non-monotonic"):
            simulator.photoelectrons_to_reflectance(np.array([10000.0, 20000.0]))

    def test_apply_convolution_uniform_input_does_not_explode(self) -> None:
        """A uniform-gray input must stay finite through the convolution path.

        A naive min/max stretch would divide by zero on uniform input and
        propagate NaN into the FFT. The method maps uniform inputs to the
        reflectance-range midpoint instead.
        """
        simulator, _img = _build_simulator(reflectance_range=np.array([0.05, 0.5]))
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        psf = simulator._get_psf_cached(gsd=3.19 / 160.0)
        true_img, blur_img = simulator.apply_convolution(uniform, psf)
        assert np.all(np.isfinite(true_img))
        assert np.all(np.isfinite(blur_img))

    def test_a_zero_degenerate_raises(self) -> None:
        """A flat sensor calibration must raise rather than divide by zero on inversion.

        When the sensor maps every reflectance to the same photoelectron
        count, the inverse is undefined. Failing fast with a clear
        ValueError is better than propagating NaN or Inf into the
        reflectance output.
        """
        simulator, _img = _build_simulator()
        fwd = cast(interpolate.interp1d, simulator._reflect_to_photoelectrons)
        ref_grid = fwd.x.copy()
        pe_const = np.full_like(ref_grid, 1000.0)
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_const)
        simulator._init_inverse_coefs(ref_grid, pe_const)
        with pytest.raises(ValueError, match="radiometric calibration is flat"):
            simulator.photoelectrons_to_reflectance(np.array([1000.0]))

    def test_pe_to_pixels_empty_input_returns_empty(self) -> None:
        """An empty input array must produce an empty output of the same shape.

        Otherwise the per-image min/max or clip-fraction division would
        crash on a zero-size array.
        """
        simulator, _img = _build_simulator()
        empty = np.empty((0, 4), dtype=np.float64)
        out = simulator.photoelectrons_to_pixels(empty)
        assert out.shape == (0, 4)
        assert out.dtype == np.float64
        assert out.size == 0

    def test_pe_to_pixels_single_element_input(self) -> None:
        """A single-element input must produce a valid one-element output.

        The clip-fraction counter divides by array size. A divisor of
        ``size - 1`` would crash on a one-element input.
        """
        simulator, _img = _build_simulator()
        pe_mid = 0.5 * (simulator._fwd_y_min + simulator._fwd_y_max)
        out = simulator.photoelectrons_to_pixels(np.array([pe_mid]))
        assert out.shape == (1,)
        assert 0.0 <= out[0] <= 255.0
        assert simulator._last_clip_fraction == (0.0, 0.0)

    @pytest.mark.parametrize("ndim", [2, 3], ids=["2D", "3D"])
    def test_apply_noise_raises_on_non_finite_input(self, ndim: int) -> None:
        """apply_noise must raise on a NaN input rather than hanging the noise kernel.

        The numba parallel fast-math kernel hangs indefinitely on
        ``np.random.poisson(NaN)``; the guard at the Python boundary fails
        fast with a diagnostic message instead.
        """
        simulator, _img = _build_simulator()
        simulator._add_noise = True
        simulator._g_noise = 1.0
        shape = (32, 32) if ndim == 2 else (32, 32, 3)
        arr = np.full(shape, 100.0, dtype=np.float64)
        arr.flat[0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite image"):
            simulator.apply_noise(arr)

    @pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3)], ids=["full_2D", "full_3D"])
    def test_apply_noise_finite_input_passes_through(self, shape: tuple[int, ...]) -> None:
        """Finite inputs must flow through the non-finite guard with shape preserved.

        The guard is a gate, not a transform: finite arrays must be
        dispatched to the noise kernel and returned with noise applied.
        A guard that short-circuited would return the input unchanged,
        silently producing noise-free output that the caller would not
        notice until downstream metrics looked wrong.
        """
        simulator, _img = _build_simulator()
        simulator._add_noise = True
        simulator._g_noise = 1.0
        arr = np.full(shape, 100.0, dtype=np.float64)
        out = simulator.apply_noise(arr)
        assert out.shape == shape
        assert np.all(np.isfinite(out))

    def test_cross_shape_simulate_image_is_stable(self) -> None:
        """One simulator must work on input frames of different shapes.

        The PSF cache and noise kernels do not capture image shape, so a
        single simulator can process arbitrary frame sizes back-to-back.
        Shape-dependent state would silently force users to build a new
        simulator per frame size — a regression that would surface as
        confusing crashes far from the source.
        """
        simulator, img = _build_simulator()
        gsd = 3.19 / 160.0
        _, blur_a, _ = simulator.simulate_image(img, gsd=gsd)
        assert np.all(np.isfinite(blur_a))
        small = img[:64, :64].copy()
        _, blur_b, _ = simulator.simulate_image(small, gsd=gsd)
        assert np.all(np.isfinite(blur_b))
        assert blur_b.shape == small.shape

    @pytest.mark.parametrize(
        ("p1", "p2"),
        [(0.0, 255.0), (10.0, 200.0)],
        ids=["default_output", "custom_output"],
    )
    def test_minmax_mode_remap(self, p1: float, p2: float) -> None:
        """mode='minmax' must stretch input min/max linearly into ``[p1, p2]``.

        A regression in the remap formula would shift an endpoint off
        ``p1`` / ``p2`` or push output values outside ``[p1, p2]``.
        """
        simulator, _img = _build_simulator()
        pe = np.linspace(1000.0, 50_000.0, 256)
        pixels = simulator.photoelectrons_to_pixels(pe, p1=p1, p2=p2, mode="minmax")
        assert pixels[0] == pytest.approx(p1, abs=1e-9)
        assert pixels[-1] == pytest.approx(p2, abs=1e-9)
        assert np.all((pixels >= p1) & (pixels <= p2))
        assert np.all(np.diff(pixels) >= 0.0)

    def test_minmax_mode_uniform_input_returns_midpoint(self) -> None:
        """Uniform input on the minmax path must map to the output-range midpoint.

        Without this, the minmax branch would divide by zero on uniform
        input. Returning the midpoint matches the apply_convolution
        convention so user code sees consistent uniform-input behavior.
        """
        simulator, _img = _build_simulator()
        pe = np.full(64, 1234.0)
        pixels = simulator.photoelectrons_to_pixels(pe, mode="minmax")
        np.testing.assert_array_equal(pixels, np.full(64, 127.5, dtype=np.float64))

    def test_minmax_mode_invalid_raises(self) -> None:
        """An unknown mode literal must raise rather than silently default.

        Without this, a typo (``mode="minmaz"``) would fall through to
        the radiometric branch and silently produce the wrong output.
        """
        simulator, _img = _build_simulator()
        with pytest.raises(ValueError, match="mode must be 'radiometric' or 'minmax'"):
            simulator.photoelectrons_to_pixels(np.array([1.0, 2.0]), mode="bogus")  # type: ignore[arg-type]

    def test_minmax_mode_works_with_use_reflectance_false(self) -> None:
        """Minmax must work even when the simulator has no reflectance forward map.

        The minmax branch reads only the input's own min/max. Routing it
        through the radiometric path (which requires a forward map) would
        crash on a ``use_reflectance=False`` simulator and break the
        documented mode-agnostic contract.
        """
        simulator, _img = _build_simulator(use_reflectance=False)
        pe = np.linspace(0.0, 32400.0, 128)
        pixels = simulator.photoelectrons_to_pixels(pe, mode="minmax")
        assert pixels[0] == pytest.approx(0.0, abs=1e-9)
        assert pixels[-1] == pytest.approx(255.0, abs=1e-9)
        assert simulator._last_clip_fraction == (0.0, 0.0)

    def test_minmax_and_radiometric_diverge_under_sensor_calibration(self) -> None:
        """The two modes must produce different output when sensors differ.

        Minmax is sensor-agnostic (same input -> same pixels across
        sensors); radiometric is sensor-calibrated (same input ->
        different pixels). Two sensors viewing the same scene must
        therefore agree under minmax and disagree under radiometric —
        confusing the two paths would erase the radiometric guarantee
        user code depends on.
        """
        simulator_a, _img = _build_simulator()
        simulator_b, _img = _build_simulator()
        # Pure-offset shift is a unit-test contrivance — a real sensor
        # recalibration would change both slope and intercept; this just
        # makes B's (A, B) differ from A's so radiometric outputs diverge.
        fwd = cast(interpolate.interp1d, simulator_b._reflect_to_photoelectrons)
        ref_b = fwd.x.copy()
        pe_b = fwd.y.copy() + (fwd.y[-1] - fwd.y[0]) * 0.5
        simulator_b._reflect_to_photoelectrons = interpolate.interp1d(ref_b, pe_b)
        simulator_b._init_inverse_coefs(ref_b, pe_b)
        pe_input = np.linspace(fwd.y[0], fwd.y[-1], 5000)
        minmax_a = simulator_a.photoelectrons_to_pixels(pe_input, mode="minmax")
        minmax_b = simulator_b.photoelectrons_to_pixels(pe_input, mode="minmax")
        np.testing.assert_array_equal(minmax_a, minmax_b)
        radio_a = simulator_a.photoelectrons_to_pixels(pe_input, mode="radiometric")
        radio_b = simulator_b.photoelectrons_to_pixels(pe_input, mode="radiometric")
        # The 0.5x range shift produces a mean pixel difference of ~30 uint8
        # levels; threshold 5.0 sits well above float-precision noise without
        # weakening the assertion.
        assert abs(radio_a.mean() - radio_b.mean()) > 5.0
