from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from syrupy.assertion import SnapshotAssertion

from pybsm import simulation

BASE_FILE_PATH = Path(__file__).parent.parent.parent
IMAGE_FILE_PATH = BASE_FILE_PATH / "examples" / "data" / "M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


class TestSimulation:
    @pytest.mark.parametrize(
        ("img", "pix_values", "refl_values"),
        [
            (np.array([]), np.array([]), np.array([])),
            (np.array([0.0]), np.array([]), np.array([])),
            (np.array([]), np.array([0.0]), np.array([])),
            (np.array([]), np.array([]), np.array([0.0])),
        ],
    )
    def test_img_to_reflectance_value_error(
        self,
        img: np.ndarray,
        pix_values: np.ndarray,
        refl_values: np.ndarray,
    ) -> None:
        """Cover cases where ValueError occurs."""
        with pytest.raises(ValueError):  # noqa: PT011
            simulation.img_to_reflectance(img, pix_values, refl_values)

    @pytest.mark.parametrize(
        ("img", "pix_values", "refl_values"),
        [
            (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        ],
    )
    def test_img_to_reflectance_nan(
        self,
        img: np.ndarray,
        pix_values: np.ndarray,
        refl_values: np.ndarray,
    ) -> None:
        """Cover cases where nan occurs."""
        output = simulation.img_to_reflectance(img, pix_values, refl_values)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("img", "pix_values", "refl_values", "expected"),
        [
            (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
            ),
            (
                np.ones((10, 10)),
                np.array([0.0, 1.0]),
                np.array([0.0, 2.0]),
                np.ones((10, 10)),
            ),
        ],
    )
    def test_img_to_reflectance(
        self,
        img: np.ndarray,
        pix_values: np.ndarray,
        refl_values: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test img_to_reflectance with normal inputs and expected outputs."""
        output = simulation.img_to_reflectance(img, pix_values, refl_values)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("w", "f", "expectation"),
        [
            (1, 1, does_not_raise()),
            (1, 0, pytest.raises(ZeroDivisionError)),
            (0, 0, pytest.raises(ZeroDivisionError)),
        ],
    )
    def test_instantaneous_FOV_ZeroDivision_error(  # noqa: N802
        self,
        w: int,
        f: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Cover cases where ValueError occurs."""
        with expectation:
            simulation.instantaneous_FOV(w, f)

    @pytest.mark.parametrize(("w", "f"), [(1, 1), (2, 1), (1, 2)])
    def test_instantaneous_FOV(self, w: int, f: int, snapshot: SnapshotAssertion) -> None:  # noqa: N802
        output = simulation.instantaneous_FOV(w, f)
        assert output == snapshot

    @pytest.mark.parametrize(
        ("otf", "noise_to_signal_power_spectrum", "expected"),
        [
            (np.array([1.0, 0.0]), 0.5, np.array([2 / 3, 0.0])),
            (np.array([0.0, 0.0]), 1.0, np.array([0.0, 0.0])),
        ],
    )
    def test_wiener_filter(self, otf: np.ndarray, noise_to_signal_power_spectrum: float, expected: np.ndarray) -> None:
        output = simulation.wiener_filter(otf, noise_to_signal_power_spectrum)
        print(output)
        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        ("otf", "noise_to_signal_power_spectrum"),
        [
            (np.array([1.0, 0.0]), 0.0),
            (np.array([0.0, 0.0]), 0.0),
        ],
    )
    def test_wiener_filter_nan(
        self,
        otf: np.ndarray,
        noise_to_signal_power_spectrum: float,
    ) -> None:
        output = simulation.wiener_filter(otf, noise_to_signal_power_spectrum)
        assert np.isnan(output).any()

    @pytest.mark.parametrize(
        ("img_file_path", "perc"),
        [
            (
                IMAGE_FILE_PATH,
                None,
            ),
            (
                IMAGE_FILE_PATH,
                [10.0, 90.0],
            ),
        ],
    )
    def test_stretch_contrast_convert_8bit(
        self,
        img_file_path: str,
        perc: list[float],
        snapshot: SnapshotAssertion,
    ) -> None:
        img = np.array(Image.open(img_file_path))
        output = [simulation.stretch_contrast_convert_8bit(img, perc)]
        assert output == snapshot

    @pytest.mark.parametrize(
        ("img_file_path", "gsd", "altitude", "true_img_file_path", "blur_img_file_path", "noisy_img_file_path"),
        [
            (
                IMAGE_FILE_PATH,
                3.19 / 160.0,
                1000,
                BASE_FILE_PATH / "tests" / "data" / "test_simulate_true_img.png",
                BASE_FILE_PATH / "tests" / "data" / "test_simulate_blur_img.png",
                BASE_FILE_PATH / "tests" / "data" / "test_simulate_noisy_img.png",
            ),
        ],
    )
    def test_simulate_image(
        self,
        img_file_path: str,
        gsd: float,
        altitude: int,
        true_img_file_path: Path,
        blur_img_file_path: Path,
        noisy_img_file_path: Path,
    ) -> None:
        img = np.array(Image.open(img_file_path))
        expected_true_img = np.array(Image.open(true_img_file_path))
        expected_blur_img = np.array(Image.open(blur_img_file_path))
        expected_noisy_img = np.array(Image.open(noisy_img_file_path))
        ref_img = simulation.RefImage(img, gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)
        true_img, blur_img, noisy_img = simulation.simulate_image(ref_img, sensor, scenario)
        true_img = simulation.stretch_contrast_convert_8bit(true_img)
        blur_img = simulation.stretch_contrast_convert_8bit(blur_img)
        noisy_img = simulation.stretch_contrast_convert_8bit(noisy_img)
        assert np.isclose(true_img.all(), expected_true_img.all())
        assert np.isclose(blur_img.all(), expected_blur_img.all())
        assert np.isclose(noisy_img.all(), expected_noisy_img.all())
