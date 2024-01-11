import numpy as np
import pytest

from pybsm import radiance
from pybsm import utils
from pybsm.simulation import Sensor
from typing import Tuple


@pytest.mark.filterwarnings("ignore:Input array")
class TestReflectance2Photoelectrons:

    @pytest.mark.parametrize("E, wx, wy, wavelengths, qe", [
        (np.array([]), 0.0, 0.0, np.array([]), np.array([])),
        (np.array([1.0]), 0.0, 0.0, np.array([]), np.array([1.0])),
        (np.array([]), 0.0, 0.0, np.array([1.0]), np.array([1.0])),
        (np.array([1.0]), 0.0, 0.0, np.array([1.0]), np.array([])),
    ])
    def test_photon_detection_rate_empty_array(
        self,
        E: np.ndarray,
        wx: float,
        wy: float,
        wavelengths: np.ndarray,
        qe: np.ndarray
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.photonDetectionRate(E, wx, wy, wavelengths, qe)
        assert output.size == 0

    @pytest.mark.parametrize("E, wx, wy, wavelengths, qe, expected", [
        (np.array([1.0]), 1.0, 1.0, np.array([1.0]), np.array([1.0]), np.array([5.03411665e+24])),
        (np.array([1.0, 1.0]), 1.0, 1.0, np.array([1.0, 1.0]), np.array([1.0, 1.0]),
            np.array([5.03411665e+24, 5.03411665e+24])),
    ])
    def test_photon_detection_rate(
        self,
        E: np.ndarray,
        wx: float,
        wy: float,
        wavelengths: np.ndarray,
        qe: np.ndarray,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.photonDetectionRate(E, wx, wy, wavelengths, qe)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("D, f, L", [
        (0.0, 0.0, np.array([])),
        (0.0, 1.0, np.array([])),
    ])
    def test_at_focal_plane_irradiance_zero_division(
        self,
        D: float,
        f: float,
        L: np.ndarray
    ) -> None:
        """
        Cover cases where ZeroDivisionError occurs
        """
        with pytest.raises(ZeroDivisionError):
            radiance.atFocalPlaneIrradiance(D, f, L)

    @pytest.mark.parametrize("D, f, L", [
        (1.0, 0.0, np.array([])),
    ])
    def test_at_focal_plane_irradiance_empty_array(
        self,
        D: float,
        f: float,
        L: np.ndarray,
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.atFocalPlaneIrradiance(D, f, L)
        assert output.size == 0

    @pytest.mark.parametrize("D, f, L, expected", [
        (1.0, 0.0, np.array([1.0]), np.array([3.14159265])),
        (1.0, 1.0, np.array([1.0]), np.array([0.62831853])),
        (1.0, 1.0, np.array([1.0, 1.0]), np.array([0.62831853, 0.62831853])),
        (1.0, 1.0, np.array([1.0, 2.0]), np.array([0.62831853, 1.25663706])),
    ])
    def test_at_focal_plane_irradiance(
        self,
        D: float,
        f: float,
        L: np.ndarray,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.atFocalPlaneIrradiance(D, f, L)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("lambda0, T", [
        (np.array([]), 0.0),
    ])
    def test_blackbody_radiance_empty_array(
        self,
        lambda0: np.ndarray,
        T: float
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.blackbodyRadiance(lambda0, T)
        assert output.size == 0

    @pytest.mark.parametrize("lambda0, T, expected", [
        (np.array([1.0]), 1.0, np.array([8.21875092e-15])),
        (np.array([1.0, 1.0]), 1.0, np.array([8.21875092e-15, 8.21875092e-15])),
    ])
    def test_blackbody_radiance(
        self,
        lambda0: np.ndarray,
        T: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.blackbodyRadiance(lambda0, T)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("wavelengths, values, newWavelengths", [
        (np.array([]), np.array([]), np.array([])),
        (np.array([1.0]), np.array([]), np.array([1.0])),
        (np.array([]), np.array([1.0]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 1.5, 2.0]), np.array([0.0, 1.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 3.0])),
    ])
    def test_resample_by_wavelength_value_error(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        newWavelengths: np.ndarray
    ) -> None:
        """
        Cover cases where ValueError occurs
        """
        with pytest.raises(ValueError):
            radiance.resampleByWavelength(wavelengths, values, newWavelengths)

    @pytest.mark.parametrize("wavelengths, values, newWavelengths", [
        (np.array([1.0]), np.array([1.0]), np.array([])),
    ])
    def test_resample_by_wavelength_empty_array(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        newWavelengths: np.ndarray
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.resampleByWavelength(wavelengths, values, newWavelengths)
        assert output.size == 0

    @pytest.mark.parametrize("wavelengths, values, newWavelengths, expected", [
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([1.0, 1.5, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 0.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0])),
    ])
    def test_resample_by_wavelength(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        newWavelengths: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.resampleByWavelength(wavelengths, values, newWavelengths)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(("wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe, otherIrradiance,"
                             "darkCurrent"), [
        (np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([]), 0.0),
        (np.array([]), np.array([]), np.array([]), 0.0, 1.0, 1.0, 1.0, np.array([]), np.array([]), 1.0),
    ])
    def test_signal_rate_zero_division(
        self,
        wavelengths: np.ndarray,
        targetRadiance: np.ndarray,
        opticalTransmission: np.ndarray,
        D: float,
        f: float,
        wx: float,
        wy: float,
        qe: np.ndarray,
        otherIrradiance: np.ndarray,
        darkCurrent: float
    ) -> None:
        """
        Cover cases where ZeroDivisionError occurs
        """
        with pytest.raises(ZeroDivisionError):
            radiance.signalRate(wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe,
                                otherIrradiance, darkCurrent)

    @pytest.mark.parametrize(("wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe, otherIrradiance,"
                             "darkCurrent, expected"), [
        (np.array([]), np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0, np.array([]), np.array([]), 1.0, 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([]), 1.0,
            1.0),
        (np.array([1.0]), np.array([1.0]), np.array([]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([1.0]), 1.0,
            1.0),
        (np.array([1.0]), np.array([]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([1.0]), 1.0,
            1.0),
    ])
    def test_signal_rate_both_arrays_empty(
        self,
        wavelengths: np.ndarray,
        targetRadiance: np.ndarray,
        opticalTransmission: np.ndarray,
        D: float,
        f: float,
        wx: float,
        wy: float,
        qe: np.ndarray,
        otherIrradiance: np.ndarray,
        darkCurrent: float,
        expected: float
    ) -> None:
        """
        Cover cases where both output arrays are empty
        """
        output = radiance.signalRate(wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe,
                                     otherIrradiance, darkCurrent)
        assert np.isclose(output[0], expected)
        assert output[1].size == 0
        assert output[2].size == 0

    @pytest.mark.parametrize(("wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe, otherIrradiance,"
                             "darkCurrent, expected"), [
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([]), np.array([1.0]), 1.0,
            (1.0, np.array([1.62831853]))),
        (np.array([]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([1.0]), 1.0,
            (1.0, np.array([1.62831853]))),
    ])
    def test_signal_rate_second_array_empty(
        self,
        wavelengths: np.ndarray,
        targetRadiance: np.ndarray,
        opticalTransmission: np.ndarray,
        D: float,
        f: float,
        wx: float,
        wy: float,
        qe: np.ndarray,
        otherIrradiance: np.ndarray,
        darkCurrent: float,
        expected: Tuple[float, np.ndarray]
    ) -> None:
        """
        Cover cases where second output array is empty
        """
        output = radiance.signalRate(wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe,
                                     otherIrradiance, darkCurrent)
        assert np.isclose(output[0], expected[0])
        assert np.isclose(output[1], expected[1]).all()
        assert output[2].size == 0

    @pytest.mark.parametrize(("wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe, otherIrradiance,"
                             "darkCurrent, expected"), [
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([1.0]), 1.0,
            (1.0, np.array([1.62831853]), np.array([8.19714543e+24]))),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0, 1.0]),
            np.array([1.0, 1.0]), 1.0,
            (1.0, np.array([1.62831853, 1.62831853]), np.array([8.19714543e+24, 8.19714543e+24]))),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0, 1.0]),
            np.array([1.0, 1.0]), 2.0,
            (2.0, np.array([1.62831853, 1.62831853]), np.array([8.19714543e+24, 8.19714543e+24]))),
    ])
    def test_signal_rate(
        self,
        wavelengths: np.ndarray,
        targetRadiance: np.ndarray,
        opticalTransmission: np.ndarray,
        D: float,
        f: float,
        wx: float,
        wy: float,
        qe: np.ndarray,
        otherIrradiance: np.ndarray,
        darkCurrent: float,
        expected: Tuple[float, np.ndarray, np.ndarray]
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.signalRate(wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe,
                                     otherIrradiance, darkCurrent)
        assert np.isclose(output[0], expected[0])
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize("wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f", [
        (np.array([]), 0.0, 0.0, 0.0, 0.0),
        (np.array([]), 1.0, 1.0, 0.0, 1.0),
    ])
    def test_coldstop_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        coldfilterTemperature: float,
        coldfilterEmissivity: float,
        D: float,
        f: float,
    ) -> None:
        """
        Cover cases where ZeroDivisionError occurs
        """
        with pytest.raises(ZeroDivisionError):
            radiance.coldstopSelfEmission(wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f)

    @pytest.mark.parametrize("wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f", [
        (np.array([]), 0.0, 0.0, 1.0, 0.0),
        (np.array([]), 1.0, 1.0, 1.0, 1.0),
    ])
    def test_coldstop_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        coldfilterTemperature: float,
        coldfilterEmissivity: float,
        D: float,
        f: float,
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.coldstopSelfEmission(wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f)
        assert output.size == 0

    @pytest.mark.parametrize("wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f, expected", [
        (np.array([1.0]), 0.0, 0.0, 1.0, 0.0, np.array([0.0])),
        (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15])),
    ])
    def test_coldstop_self_emission(
        self,
        wavelengths: np.ndarray,
        coldfilterTemperature: float,
        coldfilterEmissivity: float,
        D: float,
        f: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.coldstopSelfEmission(wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(("wavelengths, opticsTemperature, opticsEmissivity, coldfilterTransmission, D, f"), [
        (np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0),
        (np.array([]), 1.0, 1.0, 1.0, 0.0, 1.0),
    ])
    def test_optics_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        opticsTemperature: float,
        opticsEmissivity: float,
        coldfilterTransmission: float,
        D: float,
        f: float
    ) -> None:
        """
        Cover cases where ZeroDivisionError occurs
        """
        with pytest.raises(ZeroDivisionError):
            radiance.opticsSelfEmission(wavelengths, opticsTemperature, opticsEmissivity,
                                        coldfilterTransmission, D, f)

    @pytest.mark.parametrize(("wavelengths, opticsTemperature, opticsEmissivity, coldfilterTransmission, D, f"), [
        (np.array([]), 0.0, 0.0, 0.0, 1.0, 0.0),
        (np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0),
    ])
    def test_optics_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        opticsTemperature: float,
        opticsEmissivity: float,
        coldfilterTransmission: float,
        D: float,
        f: float
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.opticsSelfEmission(wavelengths, opticsTemperature, opticsEmissivity,
                                             coldfilterTransmission, D, f)
        assert output.size == 0

    @pytest.mark.parametrize(("wavelengths, opticsTemperature, opticsEmissivity, coldfilterTransmission, D, f,"
                             "expected"), [
        (np.array([1.0]), 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0])),
        (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15])),
    ])
    def test_optics_self_emission(
        self,
        wavelengths: np.ndarray,
        opticsTemperature: float,
        opticsEmissivity: float,
        coldfilterTransmission: float,
        D: float,
        f: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.opticsSelfEmission(wavelengths, opticsTemperature, opticsEmissivity,
                                             coldfilterTransmission, D, f)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("wavelengths, coldshieldTemperature, D, f", [
        (np.array([]), 0.0, 0.0, 0.0),
        (np.array([]), 1.0, 0.0, 1.0),
    ])
    def test_coldshield_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        coldshieldTemperature: float,
        D: float,
        f: float
    ) -> None:
        """
        Cover cases where ZeroDivisionError occurs
        """
        with pytest.raises(ZeroDivisionError):
            radiance.coldshieldSelfEmission(wavelengths, coldshieldTemperature, D, f)

    @pytest.mark.parametrize("wavelengths, coldshieldTemperature, D, f", [
        (np.array([]), 0.0, 1.0, 0.0),
        (np.array([]), 1.0, 1.0, 1.0),
    ])
    def test_coldshield_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        coldshieldTemperature: float,
        D: float,
        f: float
    ) -> None:
        """
        Cover cases where output is an empty array
        """
        output = radiance.coldshieldSelfEmission(wavelengths, coldshieldTemperature, D, f)
        assert output.size == 0

    @pytest.mark.parametrize("wavelengths, coldshieldTemperature, D, f, expected", [
        (np.array([1.0]), 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15])),
    ])
    def test_coldshield_self_emission(
        self,
        wavelengths: np.ndarray,
        coldshieldTemperature: float,
        D: float,
        f: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.coldshieldSelfEmission(wavelengths, coldshieldTemperature, D, f)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("atm, reflectance, temperature", [
        (np.array([]), 0.0, 0.0),
        (np.array([]), 1.0, 1.0),
    ])
    def test_total_radiance_index_error(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float
    ) -> None:
        """
        Cover cases where IndexError occurs
        """
        with pytest.raises(IndexError):
            radiance.totalRadiance(atm, reflectance, temperature)

    @pytest.mark.parametrize("atm, reflectance, temperature, expected", [
        (np.ones(shape=(6, 6)), 0.0, 0.0, np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])),
        (np.ones(shape=(6, 6)), 1.0, 1.0,
            np.array([8.66666667, 8.66666667, 8.66666667, 8.66666667, 8.66666667, 8.66666667])),
    ])
    def test_total_radiance(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.totalRadiance(atm, reflectance, temperature)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("atm, reflectance, temperature, expected", [
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), 0.0, 0.0,
            np.load('./tests/radiance/data/total_radiance_atm_zero.npy')),
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), 1.0, 1.0,
            np.load('./tests/radiance/data/total_radiance_atm_one.npy')),
    ])
    def test_total_radiance_atm(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        expected: np.ndarray
    ) -> None:
        """
        Cover cases with normal inputs, atm input, and expected outputs
        """
        output = radiance.totalRadiance(atm, reflectance, temperature)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("atm, sensor, intTime, target_temp", [
        (np.array([]), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 0.0, 0),
        (np.array([]), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 1.0, 1),
    ])
    def test_reflectance_2_photoelectrons_index_error(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        intTime: float,
        target_temp: int
    ) -> None:
        """
        Cover cases where IndexError occurs
        """
        with pytest.raises(IndexError):
            radiance.reflectance2photoelectrons(atm, sensor, intTime, target_temp)

    @pytest.mark.parametrize("atm, sensor, intTime, target_temp, expected", [
        (np.ones(shape=(6, 6)), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 0.0, 0,
            (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6)))),
        (np.ones(shape=(6, 6)), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 1.0, 300,
            (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6)))),
    ])
    def test_reflectance_2_photoelectrons(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        intTime: float,
        target_temp: int,
        expected: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Cover cases with normal inputs and expected outputs
        """
        output = radiance.reflectance2photoelectrons(atm, sensor, intTime, target_temp)
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize("atm, sensor, intTime, target_temp, expected", [
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
            1.0, 300, (np.linspace(0.0, 1.0, 100), np.load('./tests/radiance/data/reflectance_2_photoelectrons_pe.npy'),
                       np.load('./tests/radiance/data/reflectance_2_photoelectrons_spectral_weights.npy'))),
    ])
    def test_reflectance_2_photoelectrons_atm(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        intTime: float,
        target_temp: int,
        expected: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """
        Cover cases with normal inputs, atm input, and expected outputs
        """
        output = radiance.reflectance2photoelectrons(atm, sensor, intTime, target_temp)
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()
