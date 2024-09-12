import unittest.mock as mock
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

from pybsm import otf
from pybsm.simulation import Scenario, Sensor

try:
    import cv2

    is_usable = True
except ImportError:
    is_usable = False


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
class TestOTF:
    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (0.0, np.array([]), np.array([])),
            (0.0, np.array([]), np.array([1.0])),
            (1.0, np.array([]), np.array([])),
        ],
    )
    def test_coherence_diameter_value_error(
        self, lambda0: float, z_path: np.ndarray, cn2: np.ndarray
    ) -> None:
        """Cover cases where ValueError occurs."""
        with pytest.raises(ValueError):  # noqa: PT011 - This just raises a ValueError
            otf.coherence_diameter(lambda0, z_path, cn2)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (0.0, np.array([1.0]), np.array([])),
            (0.0, np.array([1.0]), np.array([0.0])),
            (0.0, np.array([1.0]), np.array([1.0])),
        ],
    )
    def test_coherence_diameter_zero_division(
        self, lambda0: float, z_path: np.ndarray, cn2: np.ndarray
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.coherence_diameter(lambda0, z_path, cn2)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (1.0, np.array([1.0]), np.array([0.0])),
            (1.0, np.array([1.0, 2.0]), np.array([0.0])),
            (1.0, np.array([1.0]), np.array([1.0])),
            (1.0, np.array([2.0]), np.array([1.0])),
        ],
    )
    def test_coherence_diameter_infinite(
        self, lambda0: float, z_path: np.ndarray, cn2: np.ndarray
    ) -> None:
        """Cover cases where infinte output occurs."""
        output = otf.coherence_diameter(lambda0, z_path, cn2)
        assert np.isinf(output)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2", "expected"),
        [
            (1.0, np.array([1.0, 2.0]), np.array([1.0]), 0.23749058343491444),
            (2.0, np.array([1.0, 2.0]), np.array([1.0]), 0.5456100850379446),
            (1.0, np.array([1.0, 2.0]), np.array([2.0]), 0.15668535178821985),
            (1.0, np.array([1.0, 2.0, 3.0]), np.array([1.0]), 0.17546491199555045),
        ],
    )
    def test_coherence_diameter(
        self, lambda0: float, z_path: np.ndarray, cn2: np.ndarray, expected: float
    ) -> None:
        """Test coherence_diameter with normal inputs and expected outputs."""
        output = otf.coherence_diameter(lambda0, z_path, cn2)
        assert np.isclose(output, expected)

    @pytest.mark.parametrize(
        ("h", "v", "cn2_at_1m"),
        [
            (np.array([]), 0.0, 0.0),
            (np.array([]), 1.0, 1.0),
        ],
    )
    def test_hufnagel_valley_turbulence_profile_empty_array(
        self, h: np.ndarray, v: float, cn2_at_1m: float
    ) -> None:
        """Test hufnagel_valley_turbulence_profile with empty input."""
        output = otf.hufnagel_valley_turbulence_profile(h, v, cn2_at_1m)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("h", "v", "cn2_at_1m", "expected"),
        [
            (np.array([1.0]), 1.0, 0.0, np.array([0.0])),
            (np.array([1.0]), 0.0, 1.0, np.array([0.9900498337491683])),
            (np.array([0.0]), 1.0, 1.0, np.array([1.0])),
            (np.array([1.0]), 1.0, 1.0, np.array([0.9900498337491683])),
            (np.array([-1.0]), -1.0, -1.0, np.array([-1.0100501670841677])),
            (np.array([1.0, 1.0]), 1.0, 0.0, np.array([0.0, 0.0])),
        ],
    )
    def test_hufnagel_valley_turbulence_profile(
        self, h: np.ndarray, v: float, cn2_at_1m: float, expected: np.ndarray
    ) -> None:
        """Test hufnagel_valley_turbulence_profile with normal inputs and expected outputs."""
        output = otf.hufnagel_valley_turbulence_profile(h, v, cn2_at_1m)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function"),
        [
            (np.array([]), np.array([]), lambda wavelengths: wavelengths),
            (np.array([]), np.array([0.0]), lambda wavelengths: wavelengths),
            (np.array([0.0]), np.array([]), lambda wavelengths: wavelengths),
            (np.array([1.0, 2.0]), np.array([1.0]), lambda wavelengths: wavelengths),
        ],
    )
    def test_weighted_by_wavelength_index_error(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        my_function: Callable,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.weighted_by_wavelength(wavelengths, weights, my_function)

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function"),
        [
            (np.array([0.0]), np.array([0.0]), lambda wavelengths: wavelengths),
            (np.array([1.0]), np.array([0.0]), lambda wavelengths: wavelengths),
            (
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                lambda wavelengths: wavelengths,
            ),
        ],
    )
    def test_weighted_by_wavelength_nan(
        self, wavelengths: np.ndarray, weights: np.ndarray, my_function: Callable
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.weighted_by_wavelength(wavelengths, weights, my_function)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function", "expected"),
        [
            (
                np.array([0.0]),
                np.array([1.0]),
                lambda wavelengths: wavelengths,
                np.array([0.0]),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                lambda wavelengths: wavelengths,
                np.array([1.0]),
            ),
            (
                np.array([1.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths,
                np.array([0.33333]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths,
                np.array([1.66666667]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths * 2,
                np.array([3.33333333]),
            ),
        ],
    )
    def test_weighted_by_wavelength(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        my_function: Callable,
        expected: np.ndarray,
    ) -> None:
        """Test weighted_by_wavelength with normal inputs and expected outputs."""
        output = otf.weighted_by_wavelength(wavelengths, weights, my_function)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_polychromatic_turbulence_OTF_zero_division(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.polychromatic_turbulence_OTF(
                u,
                v,
                wavelengths,
                weights,
                altitude,
                slant_range,
                D,
                ha_wind_speed,
                cn2_at_1m,
                int_time,
                aircraft_speed,
            )

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_polychromatic_turbulence_OTF_index_error(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.polychromatic_turbulence_OTF(
                u,
                v,
                wavelengths,
                weights,
                altitude,
                slant_range,
                D,
                ha_wind_speed,
                cn2_at_1m,
                int_time,
                aircraft_speed,
            )

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
            "expected",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.3340840371124818]),
            ),
            (
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.3340840371124818]),
            ),
            (
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.3340840371124818]),
            ),
            (
                np.array([]),
                np.array([]),
                np.array([2.0]),
                np.array([2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.7675235677237524]),
            ),
            (
                np.array([]),
                np.array([]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.62304372]),
            ),
        ],
    )
    def test_polychromatic_turbulence_OTF_first_array_empty(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
        expected: np.ndarray,
    ) -> None:
        """Test polychromatic_turbulence_OTF with empty input."""
        output = otf.polychromatic_turbulence_OTF(
            u,
            v,
            wavelengths,
            weights,
            altitude,
            slant_range,
            D,  # noqa: N803
            ha_wind_speed,
            cn2_at_1m,
            int_time,
            aircraft_speed,
        )
        assert output[0].size == 0
        assert np.isclose(output[1], expected).all()

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
            "expected",
        ),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.74665601e-09]), np.array([0.3340840371124818])),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.74665601e-09]), np.array([0.3340840371124818])),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.57584742e-05]), np.array([0.62304372])),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([9.15552002e-10]), np.array([0.11136134570416059])),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.57584742e-05, 5.10998239e-06]), np.array([0.62304372])),
            ),
            (
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.57584742e-05, 1.95347951e-06]), np.array([0.62304372])),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                (np.array([2.57584742e-05, 1.95347951e-06]), np.array([0.62304372])),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                (np.array([4.59610705e-49]), np.array([0.14605390401093207])),
            ),
        ],
    )
    def test_polychromatic_turbulence_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
        expected: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test polychromatic_turbulence_OTF with normal inputs and expected outputs."""
        output = otf.polychromatic_turbulence_OTF(
            u,
            v,
            wavelengths,
            weights,
            altitude,
            slant_range,
            D,  # noqa: N803
            ha_wind_speed,
            cn2_at_1m,
            int_time,
            aircraft_speed,
        )
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0),
        ],
    )
    def test_detector_OTF_empty_array(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, w_x: float, w_y: float, f: float
    ) -> None:
        """Test detector_OTF with empty input."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 0.0),
        ],
    )
    def test_detector_OTF_nan(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, w_x: float, w_y: float, f: float
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, np.array([1.0])),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
                1.0,
                np.array([3.89817183e-17]),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                np.array([3.89817183e-17]),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                np.array([1.51957436e-33]),
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                np.array([1.51957436e-33, 1.51957436e-33]),
            ),
        ],
    )
    def test_detector_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        f: float,
        expected: np.ndarray,
    ) -> None:
        """Test detector_OTF with normal inputs and expected outputs."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        assert np.isclose(output, expected, atol=5e-34).all()

    @pytest.mark.parametrize(
        ("u", "v", "a_x", "a_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0),
        ],
    )
    def test_drift_OTF_empty_array(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, a_x: float, a_y: float
    ) -> None:
        """Test drift_OTF with empty input."""
        output = otf.drift_OTF(u, v, a_x, a_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "a_x", "a_y", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, np.array([1.0])),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, np.array([3.89817183e-17])),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, np.array([3.89817183e-17])),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, np.array([1.51957436e-33])),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                np.array([1.51957436e-33, 1.51957436e-33]),
            ),
        ],
    )
    def test_drift_OTF(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, a_x: float, a_y: float, expected: np.ndarray
    ) -> None:
        """Test drift_OTF with normal inputs and expected outputs."""
        output = otf.drift_OTF(u, v, a_x, a_y)
        assert np.isclose(output, expected, atol=5e-34).all()

    @pytest.mark.parametrize(
        ("u", "v", "s_x", "s_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0),
        ],
    )
    def test_jitter_OTF_empty_array(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, s_x: float, s_y: float
    ) -> None:
        """Test jitter_OTF with empty input."""
        output = otf.jitter_OTF(u, v, s_x, s_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "s_x", "s_y", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, np.array([1.0])),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, np.array([2.67528799e-09])),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, np.array([2.67528799e-09])),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                np.array(
                    [
                        7.15716584e-18,
                    ]
                ),
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                np.array(
                    [
                        7.15716584e-18,
                        7.15716584e-18,
                    ]
                ),
            ),
        ],
    )
    def test_jitter_OTF(  # noqa: N802
        self, u: np.ndarray, v: np.ndarray, s_x: float, s_y: float, expected: np.ndarray
    ) -> None:
        """Test jitter_OTF with normal inputs and expected outputs."""
        output = otf.jitter_OTF(u, v, s_x, s_y)
        assert np.isclose(output, expected, atol=5e-20).all()

    def check_OTF(  # noqa: N802
        self,
        otf: otf.OTF,
        ap_OTF: np.ndarray,  # noqa: N803
        turb_OTF: np.ndarray,  # noqa: N803
        r0_band: np.ndarray,
        det_OTF: np.ndarray,  # noqa: N803
        jit_OTF: np.ndarray,  # noqa: N803
        drft_OTF: np.ndarray,  # noqa: N803
        wav_OTF: np.ndarray,  # noqa: N803
        filter_OTF: np.ndarray,  # noqa: N803
        system_OTF: np.ndarray,  # noqa: N803
    ) -> None:
        """Internal function to check if OTF object's attributes match expected values."""
        assert np.isclose(otf.ap_OTF, ap_OTF).all()
        assert np.isclose(otf.turb_OTF, turb_OTF).all()
        assert np.isclose(otf.r0_band, r0_band).all()
        assert np.isclose(otf.det_OTF, det_OTF, atol=5e-34).all()
        assert np.isclose(otf.jit_OTF, jit_OTF).all()
        assert np.isclose(otf.drft_OTF, drft_OTF).all()
        assert np.isclose(otf.wav_OTF, wav_OTF).all()
        assert np.isclose(otf.filter_OTF, filter_OTF).all()
        assert np.isclose(otf.system_OTF, system_OTF).all()

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                0.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
            ),
        ],
    )
    def test_common_OTFs_zero_division(  # noqa: N802
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.common_OTFs(
                sensor,
                scenario,
                uu,
                vv,
                mtf_wavelengths,
                mtf_weights,
                slant_range,
                int_time,
            )

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([1.0]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test_common_OTFs_index_error(  # noqa: N802
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.common_OTFs(
                sensor,
                scenario,
                uu,
                vv,
                mtf_wavelengths,
                mtf_weights,
                slant_range,
                int_time,
            )

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
            "expected",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                {
                    "ap_OTF": np.array([0.0]),
                    "turb_OTF": np.array([1.0]),
                    "r0_band": np.array([60457834.264253505]),
                    "det_OTF": np.array([1.51957436e-33]),
                    "jit_OTF": np.array([1.0]),
                    "drft_OTF": np.array([1.0]),
                    "wav_OTF": np.array([1.0]),
                    "filter_OTF": np.array([1.0]),
                    "system_OTF": np.array([0.0]),
                },
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                {
                    "ap_OTF": np.array([0.0, 0.0]),
                    "turb_OTF": np.array([1.0, 1.0]),
                    "r0_band": np.array([60457834.264253505, 60457834.264253505]),
                    "det_OTF": np.array([1.51957436e-33, 1.51957436e-33]),
                    "jit_OTF": np.array([1.0, 1.0]),
                    "drft_OTF": np.array([1.0, 1.0]),
                    "wav_OTF": np.array([1.0, 1.0]),
                    "filter_OTF": np.array([1.0, 1.0]),
                    "system_OTF": np.array([0.0, 0.0]),
                },
            ),
        ],
    )
    def test_common_OTFs(  # noqa: N802
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
        expected: Dict[str, np.ndarray],
    ) -> None:
        """Test common_OTFs with normal inputs and expected outputs."""
        output = otf.common_OTFs(
            sensor,
            scenario,
            uu,
            vv,
            mtf_wavelengths,
            mtf_weights,
            slant_range,
            int_time,
        )
        self.check_OTF(output, **expected)

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_turbulence_OTF_empty_array(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF with empty input."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 1.0),
        ],
    )
    def test_turbulence_OTF_nan(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF where output is nan."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_turbulence_OTF_inf(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF where output is inf."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert np.isinf(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, 0.0, np.array([0.0])),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 1.0, np.array([1.0])),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 0.0, np.array([1.0])),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([2.11830623, 2.11830623]),
            ),
        ],
    )
    def test_turbulence_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
        expected: np.ndarray,
    ) -> None:
        """Test turbulenceOTF with normal inputs and expected outputs."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_wind_speed_turbulence_OTF_empty_array(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Test wind_speed_turbulence_OTF with empty input."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_wind_speed_turbulence_OTF_nan(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Test wind_speed_turbulence_OTF where output is nan."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_wind_speed_turbulence_OTF_zero_division(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel", "expected"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                np.array([1.0]),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                np.array([1.0]),
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.02636412, 0.02636412]),
            ),
        ],
    )
    def test_wind_speed_turbulence_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
        expected: np.ndarray,
    ) -> None:
        """Test wind_speed_turbulence_OTF with normal inputs and expected outputs."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_wavefront_OTF_empty_array(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
    ) -> None:
        """Test wavefront_OTF with empty input."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0),
        ],
    )
    def test_wavefront_OTF_nan(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 0.0, np.array([1.0])),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                0.0,
                0.0,
                np.array([0.36787944]),
            ),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0, np.array([1.0])),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 1.0, np.array([1.0])),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                0.0,
                np.array([0.36787944]),
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                0.0,
                1.0,
                np.array([0.36787944]),
            ),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0, np.array([1.0])),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 1.0, np.array([1.0])),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([0.42119275, 0.42119275]),
            ),
        ],
    )
    def test_wavefront_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
        expected: np.ndarray,
    ) -> None:
        """Test wavefront_OTF with normal inputs and expected outputs."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([]), np.array([]), 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 1.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0),
        ],
    )
    def test_circular_aperture_OTF_empty_array(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Test circular_aperture_OTF with empty input."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0),
        ],
    )
    def test_circular_aperture_otf_zero_division(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.circular_aperture_OTF(u, v, lambda0, D, eta)

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0),
        ],
    )
    def test_circular_aperture_OTF_nan(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, np.array([0.0])),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, np.array([0.0])),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                0.0,
                np.array([0.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                0.5,
                np.array([0.0, 0.0]),
            ),
        ],
    )
    def test_circular_aperture_OTF(  # noqa: N802
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
        expected: np.ndarray,
    ) -> None:
        """Test circular_aperture_OTF with normal inputs and expected outputs."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out"),
        [
            (np.array([]), 0.0, 0.0),
            (np.array([0.0]), 0.0, 0.0),
        ],
    )
    def test_otf_to_psf_index_error(
        self, otf_value: np.ndarray, df: float, dx_out: float
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.otf_to_psf(otf_value, df, dx_out)

    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out"),
        [
            (np.ones((10, 10)), 0.0, 0.0),
            (np.ones((10, 10)), 0.0, 1.0),
            (np.ones((10, 10)), 1.0, 0.0),
        ],
    )
    def test_otf_to_psf_zero_division(
        self, otf_value: np.ndarray, df: float, dx_out: float
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.otf_to_psf(otf_value, df, dx_out)

    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out", "expected"),
        [
            (np.ones((15, 15)), 1.0, 1.0, np.array([1.0])),
            (np.ones((100, 100)), 1.0, 1.0, np.array([1.0])),
        ],
    )
    def test_otf_to_psf(
        self, otf_value: np.ndarray, df: float, dx_out: float, expected: np.ndarray
    ) -> None:
        """Test otf_to_psf with normal inputs and expected outputs."""
        output = otf.otf_to_psf(otf_value, df, dx_out)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.array([]), 1.0, 1.0),
            (np.array([1.0]), 1.0, 1.0),
        ],
    )
    def test_resample_2D_index_error(  # noqa: N802
        self, img_in: np.ndarray, dx_in: float, dx_out: float
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.ones((5, 5)), 0.0, 0.0),
            (np.ones((5, 5)), 1.0, 0.0),
        ],
    )
    def test_resample_2D_zero_division(  # noqa: N802
        self, img_in: np.ndarray, dx_in: float, dx_out: float
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.ones((5, 5)), 0.0, 1.0),
        ],
    )
    def test_resample_2D_cv2_error(  # noqa: N802
        self, img_in: np.ndarray, dx_in: float, dx_out: float
    ) -> None:
        """Cover cases where cv2.error occurs."""
        with pytest.raises(cv2.error):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out", "expected"),
        [
            (np.ones((5, 5)), 1.0, 1.0, np.ones((5, 5))),
        ],
    )
    def test_resample_2D(  # noqa: N802
        self, img_in: np.ndarray, dx_in: float, dx_out: float, expected: np.ndarray
    ) -> None:
        """Test resample_2D with normal inputs and expected outputs."""
        output = otf.resample_2D(img_in, dx_in, dx_out)
        assert np.isclose(output, expected, atol=5e-20).all()

    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov"),
        [
            (np.array([]), 0.0, 0.0, np.array([]), 0.0, 0.0),
            (np.array([]), 1.0, 0.0, np.array([]), 0.0, 0.0),
            (np.array([]), 0.0, 0.0, np.array([]), 1.0, 0.0),
            (np.array([]), 0.0, 0.0, np.array([]), 0.0, 1.0),
            (np.array([]), 1.0, 0.0, np.array([]), 1.0, 0.0),
            (np.array([]), 1.0, 0.0, np.array([]), 0.0, 1.0),
            (np.array([]), 1.0, 0.0, np.array([]), 1.0, 1.0),
            (np.ones((100, 100)), 0.0, 1.0, np.ones((100, 100)), 0.0, 0.0),
            (np.ones((100, 100)), 1.0, 1.0, np.ones((100, 100)), 1.0, 0.0),
        ],
    )
    def test_apply_otf_to_image_zero_division(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.apply_otf_to_image(ref_img, ref_gsd, ref_range, otf_value, df, ifov)

    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov"),
        [
            (np.array([]), 0.0, 1.0, np.array([]), 0.0, 0.0),
            (np.array([1.0]), 0.0, 1.0, np.array([]), 0.0, 0.0),
            (np.array([]), 0.0, 1.0, np.array([1.0]), 0.0, 0.0),
        ],
    )
    def test_apply_otf_to_image_index_error(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.apply_otf_to_image(ref_img, ref_gsd, ref_range, otf_value, df, ifov)

    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov", "expected"),
        [
            (
                np.ones((100, 100)),
                1.0,
                1.0,
                np.ones((100, 100)),
                1.0,
                1.0,
                (np.ones((100, 100)), np.array([1.0])),
            ),
        ],
    )
    def test_apply_otf_to_image(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
        expected: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test apply_otf_to_image with normal inputs and expected outputs."""
        output = otf.apply_otf_to_image(
            ref_img, ref_gsd, ref_range, otf_value, df, ifov
        )
        assert np.isclose(output[0], expected[0], atol=5e-20).all()
        assert np.isclose(output[1], expected[1], atol=5e-20).all()


@mock.patch("pybsm.otf.functional.is_usable", False)
def test_missing_deps() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        otf.resample_2D(np.ones((5, 5)), 1.0, 1.0)
