import numpy as np
import pytest

from pybsm import otf
from pybsm.simulation import Sensor, Scenario
from typing import Callable, Dict, Tuple


class TestOTF:

    @pytest.mark.parametrize("lambda0, zPath, cn2", [
        (0.0, np.array([]), np.array([])),
        (0.0, np.array([]), np.array([1.0])),
        (1.0, np.array([]), np.array([])),
    ])
    def test_coherence_diameter_value_error(
        self,
        lambda0: float,
        zPath: np.ndarray,
        cn2: np.ndarray
    ) -> None:
        with pytest.raises(ValueError):
            otf.coherenceDiameter(lambda0, zPath, cn2)

    @pytest.mark.parametrize("lambda0, zPath, cn2", [
        (0.0, np.array([1.0]), np.array([])),
        (0.0, np.array([1.0]), np.array([0.0])),
        (0.0, np.array([1.0]), np.array([1.0])),
    ])
    def test_coherence_diameter_zero_division(
        self,
        lambda0: float,
        zPath: np.ndarray,
        cn2: np.ndarray
    ) -> None:
        with pytest.raises(ZeroDivisionError):
            otf.coherenceDiameter(lambda0, zPath, cn2)

    @pytest.mark.parametrize("lambda0, zPath, cn2", [
        (1.0, np.array([1.0]), np.array([0.0])),
        (1.0, np.array([1.0, 2.0]), np.array([0.0])),
        (1.0, np.array([1.0]), np.array([1.0])),
        (1.0, np.array([2.0]), np.array([1.0])),
    ])
    def test_coherence_diameter_infinite(
        self,
        lambda0: float,
        zPath: np.ndarray,
        cn2: np.ndarray
    ) -> None:
        output = otf.coherenceDiameter(lambda0, zPath, cn2)
        assert np.isinf(output)

    @pytest.mark.parametrize("lambda0, zPath, cn2, expected", [
        (1.0, np.array([1.0, 2.0]), np.array([1.0]), 0.23749058343491444),
        (2.0, np.array([1.0, 2.0]), np.array([1.0]), 0.5456100850379446),
        (1.0, np.array([1.0, 2.0]), np.array([2.0]), 0.15668535178821985),
        (1.0, np.array([1.0, 2.0, 3.0]), np.array([1.0]), 0.17546491199555045),
    ])
    def test_coherence_diameter(
        self,
        lambda0: float,
        zPath: np.ndarray,
        cn2: np.ndarray,
        expected: float
    ) -> None:
        output = otf.coherenceDiameter(lambda0, zPath, cn2)
        assert np.isclose(output, expected)

    @pytest.mark.parametrize("h, v, cn2at1m", [
        (np.array([]), 0.0, 0.0),
        (np.array([]), 1.0, 1.0),
    ])
    def test_hufnagel_valley_turbulence_profile_empty_array(
        self,
        h: np.ndarray,
        v: float,
        cn2at1m: float
    ) -> None:
        output = otf.hufnagelValleyTurbulenceProfile(h, v, cn2at1m)
        assert output.size == 0

    @pytest.mark.parametrize("h, v, cn2at1m, expected", [
        (np.array([1.0]), 1.0, 0.0, np.array([0.0])),
        (np.array([1.0]), 0.0, 1.0, np.array([0.9900498337491683])),
        (np.array([0.0]), 1.0, 1.0, np.array([1.0])),
        (np.array([1.0]), 1.0, 1.0, np.array([0.9900498337491683])),
        (np.array([-1.0]), -1.0, -1.0, np.array([-1.0100501670841677])),
        (np.array([1.0, 1.0]), 1.0, 0.0, np.array([0.0, 0.0])),
    ])
    def test_hufnagel_valley_turbulence_profile(
        self,
        h: np.ndarray,
        v: float,
        cn2at1m: float,
        expected: np.ndarray
    ) -> None:
        output = otf.hufnagelValleyTurbulenceProfile(h, v, cn2at1m)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("wavelengths, weights, myFunction", [
        (np.array([]), np.array([]), lambda wavelengths: wavelengths),
        (np.array([1.0, 2.0]), np.array([1.0]), lambda wavelengths: wavelengths),
    ])
    def test_weighted_by_wavelength_index_error(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        myFunction: Callable,
    ) -> None:
        with pytest.raises(IndexError):
            otf.weightedByWavelength(wavelengths, weights, myFunction)

    @pytest.mark.parametrize("wavelengths, weights, myFunction", [
        (np.array([0.0]), np.array([0.0]), lambda wavelengths: wavelengths),
        (np.array([1.0]), np.array([0.0]), lambda wavelengths: wavelengths),
        (np.array([1.0, 1.0]), np.array([0.0, 0.0]), lambda wavelengths: wavelengths),
    ])
    def test_weighted_by_wavelength_nan(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        myFunction: Callable
    ) -> None:
        output = otf.weightedByWavelength(wavelengths, weights, myFunction)
        assert np.isnan(output).all()

    @pytest.mark.parametrize("wavelengths, weights, myFunction, expected", [
        (np.array([0.0]), np.array([1.0]), lambda wavelengths: wavelengths, np.array([0.0])),
        (np.array([1.0]), np.array([1.0]), lambda wavelengths: wavelengths, np.array([1.0])),
        (np.array([1.0]), np.array([1.0, 2.0]), lambda wavelengths: wavelengths, np.array([0.33333])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), lambda wavelengths: wavelengths, np.array([1.66666667])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), lambda wavelengths: wavelengths * 2, np.array([3.33333333])),
    ])
    def test_weighted_by_wavelength(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        myFunction: Callable,
        expected: np.ndarray
    ) -> None:
        output = otf.weightedByWavelength(wavelengths, weights, myFunction)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed, cn2at1m,"
                             "intTime, aircraftSpeed",
                             [
                                (np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0),
                                (np.array([]), np.array([]), np.array([]), np.array([]), 1.0, 0.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0),
                             ])
    def test_polychromatic_turbulence_OTF_zero_division(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slantRange: float,
        D: float,
        haWindspeed: float,
        cn2at1m: float,
        intTime: float,
        aircraftSpeed: float
    ) -> None:
        with pytest.raises(ZeroDivisionError):
            otf.polychromaticTurbulenceOTF(u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed,
                                           cn2at1m, intTime, aircraftSpeed)

    @pytest.mark.parametrize("u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed, cn2at1m, intTime,"
                             "aircraftSpeed",
                             [
                                (np.array([]), np.array([]), np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0),
                                (np.array([1.0]), np.array([1.0]), np.array([]), np.array([1.0]), 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0),
                                (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([]), 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0),
                                (np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]), 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0)
                             ])
    def test_polychromatic_turbulence_OTF_index_error(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slantRange: float,
        D: float,
        haWindspeed: float,
        cn2at1m: float,
        intTime: float,
        aircraftSpeed: float
    ) -> None:
        with pytest.raises(IndexError):
            otf.polychromaticTurbulenceOTF(u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed,
                                           cn2at1m, intTime, aircraftSpeed)

    @pytest.mark.parametrize("u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed, cn2at1m, intTime,"
                             "aircraftSpeed, expected",
                             [
                                (np.array([]), np.array([]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, np.array([0.3340840371124818])),
                                (np.array([1.0]), np.array([]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, np.array([0.3340840371124818])),
                                (np.array([]), np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, np.array([0.3340840371124818])),
                                (np.array([]), np.array([]), np.array([2.0]), np.array([2.0]), 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, np.array([0.7675235677237524])),
                                (np.array([]), np.array([]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, np.array([0.62304372])),
                             ])
    def test_polychromatic_turbulence_OTF_first_array_empty(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slantRange: float,
        D: float,
        haWindspeed: float,
        cn2at1m: float,
        intTime: float,
        aircraftSpeed: float,
        expected: np.ndarray
    ) -> None:
        output = otf.polychromaticTurbulenceOTF(u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed,
                                                cn2at1m, intTime, aircraftSpeed)
        assert output[0].size == 0
        assert np.isclose(output[1], expected).all()

    @pytest.mark.parametrize("u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed, cn2at1m, intTime,"
                             "aircraftSpeed, expected",
                             [
                                (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, (np.array([2.74665601e-09]), np.array([0.3340840371124818]))),
                                (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    (np.array([2.74665601e-09]), np.array([0.3340840371124818]))),
                                (np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    (np.array([2.57584742e-05]), np.array([0.62304372]))),
                                (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]), 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0,
                                    (np.array([9.15552002e-10]), np.array([0.11136134570416059]))),
                                (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    (np.array([2.57584742e-05, 5.10998239e-06]), np.array([0.62304372]))),
                                (np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    (np.array([2.57584742e-05, 1.95347951e-06]), np.array([0.62304372]))),
                                (np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    (np.array([2.57584742e-05, 1.95347951e-06]), np.array([0.62304372]))),
                                (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0,
                                    (np.array([4.59610705e-49]), np.array([0.14605390401093207]))),
                             ])
    def test_polychromatic_turbulence_OTF(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slantRange: float,
        D: float,
        haWindspeed: float,
        cn2at1m: float,
        intTime: float,
        aircraftSpeed: float,
        expected: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        output = otf.polychromaticTurbulenceOTF(u, v, wavelengths, weights, altitude, slantRange, D, haWindspeed,
                                                cn2at1m, intTime, aircraftSpeed)
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()

    @pytest.mark.parametrize("u, v, wx, wy, f", [
        (np.array([]), np.array([]), 0.0, 0.0, 0.0),
        (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0),
        (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0),
        (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0),
    ])
    def test_detector_OTF_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wx: float,
        wy: float,
        f: float
    ) -> None:
        output = otf.detectorOTF(u, v, wx, wy, f)
        assert output.size == 0

    @pytest.mark.parametrize("u, v, wx, wy, f", [
        (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0),
        (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 0.0),
    ])
    def test_detector_OTF_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wx: float,
        wy: float,
        f: float
    ) -> None:
        output = otf.detectorOTF(u, v, wx, wy, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize("u, v, wx, wy, f, expected", [
        (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, np.array([1.0])),
        (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, np.array([3.89817183e-17])),
        (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, np.array([3.89817183e-17])),
        (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, np.array([1.51957436e-33])),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, 1.0, np.array([1.51957436e-33, 1.51957436e-33])),
    ])
    def test_detector_OTF(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wx: float,
        wy: float,
        f: float,
        expected: np.ndarray
    ) -> None:
        output = otf.detectorOTF(u, v, wx, wy, f)
        assert np.isclose(output, expected, atol=5e-34).all()

    @pytest.mark.parametrize("u, v, ax, ay", [
        (np.array([]), np.array([]), 0.0, 0.0),
        (np.array([1.0]), np.array([]), 0.0, 0.0),
        (np.array([]), np.array([1.0]), 0.0, 0.0),
        (np.array([]), np.array([]), 1.0, 1.0),
    ])
    def test_drift_OTF_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        ax: float,
        ay: float
    ) -> None:
        output = otf.driftOTF(u, v, ax, ay)
        assert output.size == 0

    @pytest.mark.parametrize("u, v, ax, ay, expected", [
        (np.array([1.0]), np.array([1.0]), 0.0, 0.0, np.array([1.0])),
        (np.array([1.0]), np.array([1.0]), 1.0, 0.0, np.array([3.89817183e-17])),
        (np.array([1.0]), np.array([1.0]), 0.0, 1.0, np.array([3.89817183e-17])),
        (np.array([1.0]), np.array([1.0]), 1.0, 1.0, np.array([1.51957436e-33])),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, np.array([1.51957436e-33, 1.51957436e-33])),
    ])
    def test_drift_OTF(
        self,
        u: np.ndarray,
        v: np.ndarray,
        ax: float,
        ay: float,
        expected: np.ndarray
    ) -> None:
        output = otf.driftOTF(u, v, ax, ay)
        assert np.isclose(output, expected, atol=5e-34).all()

    @pytest.mark.parametrize("u, v, sx, sy", [
        (np.array([]), np.array([]), 0.0, 0.0),
        (np.array([1.0]), np.array([]), 0.0, 0.0),
        (np.array([]), np.array([1.0]), 0.0, 0.0),
        (np.array([]), np.array([]), 1.0, 1.0),
    ])
    def test_jitter_OTF_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        sx: float,
        sy: float
    ) -> None:
        output = otf.jitterOTF(u, v, sx, sy)
        assert output.size == 0

    @pytest.mark.parametrize("u, v, sx, sy, expected", [
        (np.array([1.0]), np.array([1.0]), 0.0, 0.0, np.array([1.0])),
        (np.array([1.0]), np.array([1.0]), 1.0, 0.0, np.array([2.67528799e-09])),
        (np.array([1.0]), np.array([1.0]), 0.0, 1.0, np.array([2.67528799e-09])),
        (np.array([1.0]), np.array([1.0]), 1.0, 1.0, np.array([7.15716584e-18,])),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, np.array([7.15716584e-18, 7.15716584e-18,])),
    ])
    def test_jitter_OTF(
        self,
        u: np.ndarray,
        v: np.ndarray,
        sx: float,
        sy: float,
        expected: np.ndarray
    ) -> None:
        output = otf.jitterOTF(u, v, sx, sy)
        assert np.isclose(output, expected, atol=5e-20).all()

    def check_OTF(
        self,
        otf: otf.OTF,
        apOTF: np.ndarray,
        turbOTF: np.ndarray,
        r0band: np.ndarray,
        detOTF: np.ndarray,
        jitOTF: np.ndarray,
        drftOTF: np.ndarray,
        wavOTF: np.ndarray,
        filterOTF: np.ndarray,
        systemOTF: np.ndarray
    ) -> None:
        assert np.isclose(otf.apOTF, apOTF).all()
        assert np.isclose(otf.turbOTF, turbOTF).all()
        assert np.isclose(otf.r0band, r0band).all()
        assert np.isclose(otf.detOTF, detOTF, atol=5e-34).all()
        assert np.isclose(otf.jitOTF, jitOTF).all()
        assert np.isclose(otf.drftOTF, drftOTF).all()
        assert np.isclose(otf.wavOTF, wavOTF).all()
        assert np.isclose(otf.filterOTF, filterOTF).all()
        assert np.isclose(otf.systemOTF, systemOTF).all()

    @pytest.mark.parametrize("sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange,"
                             "intTime",
                             [
                                (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                                    Scenario('test_scenario', 1, 1.0, 1.0), np.array([1.0]),
                                    np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.0, 0.0),
                                (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                                    Scenario('test_scenario', 1, 1.0, 1.0), np.array([1.0]),
                                    np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.0, 1.0),
                             ])
    def test_common_OTFs_zero_division(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtfwavelengths: np.ndarray,
        mtfweights: np.ndarray,
        slantRange: float,
        intTime: float
    ) -> None:
        with pytest.raises(ZeroDivisionError):
            otf.commonOTFs(sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange, intTime)

    @pytest.mark.parametrize("sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange,"
                             "intTime",
                             [
                                (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                                    Scenario('test_scenario', 1, 1.0, 1.0), np.array([1.0]),
                                    np.array([]), np.array([]), np.array([]), 1.0, 1.0),
                                (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                                    Scenario('test_scenario', 1, 1.0, 1.0), np.array([1.0]),
                                    np.array([1.0]), np.array([1.0]), np.array([]), 1.0, 1.0),
                                (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                                    Scenario('test_scenario', 1, 1.0, 1.0), np.array([1.0]),
                                    np.array([1.0]), np.array([]), np.array([1.0]), 1.0, 1.0),
                             ])
    def test_common_OTFs_index_error(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtfwavelengths: np.ndarray,
        mtfweights: np.ndarray,
        slantRange: float,
        intTime: float
    ) -> None:
        with pytest.raises(IndexError):
            otf.commonOTFs(sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange, intTime)

    @pytest.mark.parametrize("sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange, intTime, expected", [
        (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), Scenario('test_scenario', 1, 1.0, 1.0),
            np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0,
            {
                'apOTF': np.array([0.0]),
                'turbOTF': np.array([1.0]),
                'r0band': np.array([60457834.264253505]),
                'detOTF': np.array([1.51957436e-33]),
                'jitOTF': np.array([1.0]),
                'drftOTF': np.array([1.0]),
                'wavOTF': np.array([1.0]),
                'filterOTF': np.array([1.0]),
                'systemOTF': np.array([0.0])
            }),
        (Sensor('test_scene', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), Scenario('test_scenario', 1, 1.0, 1.0),
            np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0,
            {
                'apOTF': np.array([0.0, 0.0]),
                'turbOTF': np.array([1.0, 1.0]),
                'r0band': np.array([60457834.264253505, 60457834.264253505]),
                'detOTF': np.array([1.51957436e-33, 1.51957436e-33]),
                'jitOTF': np.array([1.0, 1.0]),
                'drftOTF': np.array([1.0, 1.0]),
                'wavOTF': np.array([1.0, 1.0]),
                'filterOTF': np.array([1.0, 1.0]),
                'systemOTF': np.array([0.0, 0.0])
            }),
    ])
    def test_common_OTFs(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtfwavelengths: np.ndarray,
        mtfweights: np.ndarray,
        slantRange: float,
        intTime: float,
        expected: Dict[str, np.ndarray]
    ) -> None:
        output = otf.commonOTFs(sensor, scenario, uu, vv, mtfwavelengths, mtfweights, slantRange, intTime)
        self.check_OTF(output, **expected)
