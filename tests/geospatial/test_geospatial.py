import numpy as np
import pytest

from pybsm import geospatial


class TestGeospatial:

    @pytest.mark.parametrize("hTarget, hSensor, slantRange", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ])
    def test_nadir_angle_zero_division(
        self,
        hTarget: float,
        hSensor: float,
        slantRange: float
    ) -> None:
        """
        Cover cases where ZeroDivision occurs
        """
        with pytest.raises(ZeroDivisionError):
            geospatial.nadirAngle(hTarget, hSensor, slantRange)

    @pytest.mark.parametrize("hTarget, hSensor, slantRange, expected", [
        (1.0, 0.0, 1.0, 3.141592653589793),
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.5707962484024436),
    ])
    def test_nadir_angle(
        self,
        hTarget: float,
        hSensor: float,
        slantRange: float,
        expected: float
    ) -> None:
        """
        Test nadirAngle with normal inputs and expected outputs
        """
        output = geospatial.nadirAngle(hTarget, hSensor, slantRange)
        assert np.isclose(output, expected)

    @pytest.mark.parametrize("hTarget, hSensor, slantRange", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ])
    def test_altitude_along_zero_division(
        self,
        hTarget: float,
        hSensor: float,
        slantRange: float
    ) -> None:
        """
        Cover cases where ZeroDivision occurs
        """
        with pytest.raises(ZeroDivisionError):
            geospatial.altitudeAlongSlantPath(hTarget, hSensor, slantRange)

    @pytest.mark.parametrize("hTarget, hSensor, slantRange, expected", [
        (1.0, 0.0, 1.0, np.load('./tests/geospatial/data/altitude_1_0_1.npy')),
        (0.0, 1.0, 1.0, np.load('./tests/geospatial/data/altitude_0_1_1.npy')),
        (1.0, 1.0, 1.0, np.ones(10000)),
    ])
    def test_altitude_along_slant_path(
        self,
        hTarget: float,
        hSensor: float,
        slantRange: float,
        expected: np.ndarray
    ) -> None:
        """
        Test altitudeAlongSlantPath with normal inputs and expected outputs
        """
        output = geospatial.altitudeAlongSlantPath(hTarget, hSensor, slantRange)
        assert np.isclose(output[0], np.linspace(0.0, 1.0, 10000)).all()
        assert np.isclose(output[1], expected).all()
