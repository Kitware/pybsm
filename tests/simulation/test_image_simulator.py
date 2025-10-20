from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
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
