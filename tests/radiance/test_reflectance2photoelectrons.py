import numpy as np
import pytest

from pybsm import radiance
from pybsm import utils
from pybsm.simulation import Sensor
from typing import ContextManager, Tuple
from contextlib import nullcontext as does_not_raise


class TestReflectance2Photoelectrons:

    @pytest.mark.parametrize("E, wx, wy, wavelengths, qe, expected, expectation", [
        (np.array([]), 0.0, 0.0, np.array([]), np.array([]), np.array([]), does_not_raise()),
        (np.array([1.0]), 1.0, 1.0, np.array([1.0]), np.array([1.0]), np.array([5.03411665e+24]), does_not_raise()),
        (np.array([1.0, 1.0]), 1.0, 1.0, np.array([1.0, 1.0]), np.array([1.0, 1.0]),
            np.array([5.03411665e+24, 5.03411665e+24]), does_not_raise()),
    ])
    def test_photon_detection_rate(
        self,
        E: np.ndarray,
        wx: float,
        wy: float,
        wavelengths: np.ndarray,
        qe: np.ndarray,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        with expectation:
            output = radiance.photonDetectionRate(E, wx, wy, wavelengths, qe)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("D, f, L, expected, expectation", [
        (0.0, 0.0, np.array([]), np.array([]), pytest.raises(ZeroDivisionError)),
        (1.0, 1.0, np.array([1.0]), np.array([0.62831853]), does_not_raise()),
        (1.0, 1.0, np.array([1.0, 1.0]), np.array([0.62831853, 0.62831853]), does_not_raise()),
    ])
    def test_at_focal_plane_irradiance(
        self,
        D: float,
        f: float,
        L: np.ndarray,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.atFocalPlaneIrradiance(D, f, L)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("lambda0, T, expected, expectation", [
        (np.array([]), 0.0, np.array([]), does_not_raise()),
        (np.array([1.0]), 1.0, np.array([8.21875092e-15]), does_not_raise()),
        (np.array([1.0, 1.0]), 1.0, np.array([8.21875092e-15, 8.21875092e-15]), does_not_raise()),
    ])
    def test_blackbody_radiance(
        self,
        lambda0: np.ndarray,
        T: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        with expectation:
            output = radiance.blackbodyRadiance(lambda0, T)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("wavelengths, values, newWavelengths, expected, expectation", [
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), does_not_raise()),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([1.0, 1.5, 2.0]), np.array([1.0, 2.0, 3.0]),
            does_not_raise()),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 0.0]),
            does_not_raise()),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]),
            does_not_raise()),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]),
            does_not_raise()),
        (np.array([1.0, 2.0]), np.array([1.0, 1.5, 2.0]), np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]),
            pytest.raises(ValueError)),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]),
            pytest.raises(ValueError)),
    ])
    def test_resample_by_wavelength(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        newWavelengths: np.ndarray,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.resampleByWavelength(wavelengths, values, newWavelengths)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(("wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe, otherIrradiance,"
                             "darkCurrent, expected, expectation"), [
        (np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([]), 0.0,
            (0.0, np.array([]), np.array([])), pytest.raises(ZeroDivisionError)),
        (np.array([]), np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0, np.array([]), np.array([]), 1.0,
            (1.0, np.array([]), np.array([])), does_not_raise()),
        (np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0]), np.array([1.0]), 1.0,
            (1.0, np.array([1.62831853]), np.array([8.19714543e+24])), does_not_raise()),
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, np.array([1.0, 1.0]),
            np.array([1.0, 1.0]), 1.0,
            (1.0, np.array([1.62831853, 1.62831853]), np.array([8.19714543e+24, 8.19714543e+24])),
            does_not_raise()),
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
        expected: Tuple[float, np.ndarray, np.ndarray],
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.signalRate(wavelengths, targetRadiance, opticalTransmission, D, f, wx, wy, qe,
                                         otherIrradiance, darkCurrent)
        if output is not None:
            assert np.isclose(output[0], expected[0])
            assert np.isclose(output[1], expected[1]).all()
            assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize("wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f, expected, expectation", [
        (np.array([]), 0.0, 0.0, 0.0, 0.0, np.array([]), pytest.raises(ZeroDivisionError)),
        (np.array([]), 1.0, 1.0, 1.0, 1.0, np.array([]), does_not_raise()),
        (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15]), does_not_raise()),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15]), does_not_raise()),
    ])
    def test_coldstop_self_emission(
        self,
        wavelengths: np.ndarray,
        coldfilterTemperature: float,
        coldfilterEmissivity: float,
        D: float,
        f: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.coldstopSelfEmission(wavelengths, coldfilterTemperature, coldfilterEmissivity, D, f)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(("wavelengths, opticsTemperature, opticsEmissivity, coldfilterTransmission, D, f,"
                             "expected, expectation"), [
        (np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), pytest.raises(ZeroDivisionError)),
        (np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([]), does_not_raise()),
        (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15]), does_not_raise()),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15]), does_not_raise()),
    ])
    def test_optics_self_emission(
        self,
        wavelengths: np.ndarray,
        opticsTemperature: float,
        opticsEmissivity: float,
        coldfilterTransmission: float,
        D: float,
        f: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.opticsSelfEmission(wavelengths, opticsTemperature, opticsEmissivity,
                                                 coldfilterTransmission, D, f)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("wavelengths, coldshieldTemperature, D, f, expected, expectation", [
        (np.array([]), 0.0, 0.0, 0.0, np.array([]), pytest.raises(ZeroDivisionError)),
        (np.array([]), 1.0, 1.0, 1.0, np.array([]), does_not_raise()),
        (np.array([1.0]), 1.0, 1.0, 1.0, np.array([5.16399351e-15]), does_not_raise()),
        (np.array([1.0, 1.0]), 1.0, 1.0, 1.0, np.array([5.16399351e-15, 5.16399351e-15]), does_not_raise()),
    ])
    def test_coldshield_self_emission(
        self,
        wavelengths: np.ndarray,
        coldshieldTemperature: float,
        D: float,
        f: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.coldshieldSelfEmission(wavelengths, coldshieldTemperature, D, f)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("atm, reflectance, temperature, expected, expectation", [
        (np.array([]), 0.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (np.ones(shape=(6, 6)), 0.0, 0.0, np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]), does_not_raise()),
        (np.ones(shape=(6, 6)), 1.0, 1.0,
            np.array([8.66666667, 8.66666667, 8.66666667, 8.66666667, 8.66666667, 8.66666667]), does_not_raise()),
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), 0.0, 0.0,
            np.load('./tests/radiance/data/total_radiance_atm_zero.npy'), does_not_raise()),
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), 1.0, 1.0,
            np.load('./tests/radiance/data/total_radiance_atm_one.npy'), does_not_raise()),
    ])
    def test_total_radiance(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.totalRadiance(atm, reflectance, temperature)
        if output is not None:
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize("atm, sensor, intTime, target_temp, expected, expectation", [
        (np.array([]), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 0.0, 0,
            (np.array([]), np.array([]), np.array([])), pytest.raises(IndexError)),
        (np.ones(shape=(6, 6)), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 0.0, 0,
            (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6))), does_not_raise()),
        (np.ones(shape=(6, 6)), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 1.0, 300,
            (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6))), does_not_raise()),
        (utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), Sensor('Test', 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
            1.0, 300, (np.linspace(0.0, 1.0, 100), np.load('./tests/radiance/data/reflectance_2_photoelectrons_pe.npy'),
                       np.load('./tests/radiance/data/reflectance_2_photoelectrons_spectral_weights.npy')),
            does_not_raise()),
    ])
    def test_reflectance_2_photoelectrons(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        intTime: float,
        target_temp: int,
        expected: Tuple[np.ndarray, np.ndarray, np.ndarray],
        expectation: ContextManager
    ) -> None:
        output = None
        with expectation:
            output = radiance.reflectance2photoelectrons(atm, sensor, intTime, target_temp)
        if output is not None:
            assert np.isclose(output[0], expected[0]).all()
            assert np.isclose(output[1], expected[1]).all()
            assert np.isclose(output[2], expected[2]).all()
