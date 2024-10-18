import numpy as np
import pytest

from pybsm import simulation


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
