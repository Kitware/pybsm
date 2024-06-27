import numpy as np
import pytest

from pybsm import geospatial


class TestGeospatial:
    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range"),
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
    )
    def test_nadir_angle_zero_division(
        self, h_target: float, h_sensor: float, slant_range: float
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            geospatial.nadir_angle(h_target, h_sensor, slant_range)

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range", "expected"),
        [
            (1.0, 0.0, 1.0, 3.141592653589793),
            (0.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.5707962484024436),
        ],
    )
    def test_nadir_angle(
        self, h_target: float, h_sensor: float, slant_range: float, expected: float
    ) -> None:
        """Test nadir_angle with normal inputs and expected outputs."""
        output = geospatial.nadir_angle(h_target, h_sensor, slant_range)
        assert np.isclose(output, expected)

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range"),
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
    )
    def test_altitude_along_zero_division(
        self, h_target: float, h_sensor: float, slant_range: float
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            geospatial.altitude_along_slant_path(h_target, h_sensor, slant_range)

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range", "expected"),
        [
            (1.0, 0.0, 1.0, np.load("./tests/geospatial/data/altitude_1_0_1.npy")),
            (0.0, 1.0, 1.0, np.load("./tests/geospatial/data/altitude_0_1_1.npy")),
            (1.0, 1.0, 1.0, np.ones(10000)),
        ],
    )
    def test_altitude_along_slant_path(
        self, h_target: float, h_sensor: float, slant_range: float, expected: np.ndarray
    ) -> None:
        """Test altitude_along_slant_path with normal inputs and expected outputs."""
        output = geospatial.altitude_along_slant_path(h_target, h_sensor, slant_range)
        assert np.isclose(output[0], np.linspace(0.0, 1.0, 10000)).all()
        assert np.isclose(output[1], expected).all()
