from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import radiance, utils
from pybsm.simulation import Sensor


@pytest.mark.filterwarnings("ignore:Input array")
class TestReflectanceToPhotoelectrons:
    @pytest.mark.parametrize(
        ("E", "w_x", "w_y", "wavelengths", "qe"),
        [
            (np.array([]), 0.0, 0.0, np.array([]), np.array([])),
            (np.array([1.0]), 0.0, 0.0, np.array([]), np.array([1.0])),
            (np.array([]), 0.0, 0.0, np.array([1.0]), np.array([1.0])),
            (np.array([1.0]), 0.0, 0.0, np.array([1.0]), np.array([])),
        ],
    )
    def test_photon_detection_rate_empty_array(
        self,
        E: np.ndarray,  # noqa: N803
        w_x: float,
        w_y: float,
        wavelengths: np.ndarray,
        qe: np.ndarray,
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.photon_detection_rate(E, w_x, w_y, wavelengths, qe)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("E", "w_x", "w_y", "wavelengths", "qe", "expected"),
        [
            (
                np.array([1.0]),
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
                np.array([5.03411665e24]),
            ),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([5.03411665e24, 5.03411665e24]),
            ),
        ],
    )
    def test_photon_detection_rate(
        self,
        E: np.ndarray,  # noqa: N803
        w_x: float,
        w_y: float,
        wavelengths: np.ndarray,
        qe: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.photon_detection_rate(E, w_x, w_y, wavelengths, qe)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("D", "f", "L"),
        [
            (0.0, 0.0, np.array([])),
            (0.0, 1.0, np.array([])),
        ],
    )
    def test_at_focal_plane_irradiance_zero_division(
        self,
        D: float,  # noqa: N803
        f: float,
        L: np.ndarray,  # noqa: N803
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            radiance.at_focal_plane_irradiance(D, f, L)

    @pytest.mark.parametrize(
        ("D", "f", "L"),
        [
            (1.0, 0.0, np.array([])),
        ],
    )
    def test_at_focal_plane_irradiance_empty_array(
        self,
        D: float,  # noqa: N803
        f: float,
        L: np.ndarray,  # noqa: N803
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.at_focal_plane_irradiance(D, f, L)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("D", "f", "L", "expected"),
        [
            (1.0, 0.0, np.array([1.0]), np.array([3.14159265])),
            (1.0, 1.0, np.array([1.0]), np.array([0.62831853])),
            (1.0, 1.0, np.array([1.0, 1.0]), np.array([0.62831853, 0.62831853])),
            (1.0, 1.0, np.array([1.0, 2.0]), np.array([0.62831853, 1.25663706])),
        ],
    )
    def test_at_focal_plane_irradiance(
        self,
        D: float,  # noqa: N803
        f: float,
        L: np.ndarray,  # noqa: N803
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.at_focal_plane_irradiance(D, f, L)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("lambda0", "T"),
        [
            (np.array([]), 0.0),
        ],
    )
    def test_blackbody_radiance_empty_array(
        self,
        lambda0: np.ndarray,
        T: float,  # noqa: N803
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.blackbody_radiance(lambda0, T)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("lambda0", "T", "expected"),
        [
            (np.array([1.0]), 1.0, np.array([8.21875092e-15])),
            (np.array([1.0, 1.0]), 1.0, np.array([8.21875092e-15, 8.21875092e-15])),
        ],
    )
    def test_blackbody_radiance(
        self,
        lambda0: np.ndarray,
        T: float,  # noqa: N803
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.blackbody_radiance(lambda0, T)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("wavelengths", "values", "new_wavelengths"),
        [
            (np.array([]), np.array([]), np.array([])),
            (np.array([1.0]), np.array([]), np.array([1.0])),
            (np.array([]), np.array([1.0]), np.array([1.0])),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 1.5, 2.0]),
                np.array([0.0, 1.0, 3.0]),
            ),
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0, 3.0]),
            ),
        ],
    )
    def test_resample_by_wavelength_value_error(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        new_wavelengths: np.ndarray,
    ) -> None:
        """Cover cases where ValueError occurs."""
        with pytest.raises(ValueError):  # noqa: PT011
            radiance.resample_by_wavelength(wavelengths, values, new_wavelengths)

    @pytest.mark.parametrize(
        ("wavelengths", "values", "new_wavelengths"),
        [
            (np.array([1.0]), np.array([1.0]), np.array([])),
        ],
    )
    def test_resample_by_wavelength_empty_array(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        new_wavelengths: np.ndarray,
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.resample_by_wavelength(wavelengths, values, new_wavelengths)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("wavelengths", "values", "new_wavelengths", "expected"),
        [
            (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 3.0]),
                np.array([1.0, 1.5, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 0.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0, 3.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
        ],
    )
    def test_resample_by_wavelength(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        new_wavelengths: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.resample_by_wavelength(wavelengths, values, new_wavelengths)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "target_radiance",
            "optical_transmission",
            "D",
            "f",
            "w_x",
            "w_y",
            "qe",
            "other_irradiance",
            "dark_current",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                0.0,
                0.0,
                0.0,
                0.0,
                np.array([]),
                np.array([]),
                0.0,
            ),
            (
                np.array([]),
                np.array([]),
                np.array([]),
                0.0,
                1.0,
                1.0,
                1.0,
                np.array([]),
                np.array([]),
                1.0,
            ),
        ],
    )
    def test_signal_rate_zero_division(
        self,
        wavelengths: np.ndarray,
        target_radiance: np.ndarray,
        optical_transmission: np.ndarray,
        D: float,  # noqa: N803
        f: float,
        w_x: float,
        w_y: float,
        qe: np.ndarray,
        other_irradiance: np.ndarray,
        dark_current: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            radiance.signal_rate(
                wavelengths,
                target_radiance,
                optical_transmission,
                D,
                f,
                w_x,
                w_y,
                qe,
                other_irradiance,
                dark_current,
            )

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "target_radiance",
            "optical_transmission",
            "D",
            "f",
            "w_x",
            "w_y",
            "qe",
            "other_irradiance",
            "dark_current",
            "expected",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test_signal_rate_both_arrays_empty(
        self,
        wavelengths: np.ndarray,
        target_radiance: np.ndarray,
        optical_transmission: np.ndarray,
        D: float,  # noqa: N803
        f: float,
        w_x: float,
        w_y: float,
        qe: np.ndarray,
        other_irradiance: np.ndarray,
        dark_current: float,
        expected: float,
    ) -> None:
        """Cover cases where both output arrays are empty."""
        output = radiance.signal_rate(
            wavelengths,
            target_radiance,
            optical_transmission,
            D,
            f,
            w_x,
            w_y,
            qe,
            other_irradiance,
            dark_current,
        )
        assert np.isclose(output[0], expected)
        assert output[1].size == 0
        assert output[2].size == 0

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "target_radiance",
            "optical_transmission",
            "D",
            "f",
            "w_x",
            "w_y",
            "qe",
            "other_irradiance",
            "dark_current",
            "expected",
        ),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([]),
                np.array([1.0]),
                1.0,
                (1.0, np.array([1.62831853])),
            ),
            (
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                (1.0, np.array([1.62831853])),
            ),
        ],
    )
    def test_signal_rate_second_array_empty(
        self,
        wavelengths: np.ndarray,
        target_radiance: np.ndarray,
        optical_transmission: np.ndarray,
        D: float,  # noqa: N803
        f: float,
        w_x: float,
        w_y: float,
        qe: np.ndarray,
        other_irradiance: np.ndarray,
        dark_current: float,
        expected: tuple[float, np.ndarray],
    ) -> None:
        """Cover cases where second output array is empty."""
        output = radiance.signal_rate(
            wavelengths,
            target_radiance,
            optical_transmission,
            D,
            f,
            w_x,
            w_y,
            qe,
            other_irradiance,
            dark_current,
        )
        assert np.isclose(output[0], expected[0])
        assert np.isclose(output[1], expected[1]).all()
        assert output[2].size == 0

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "target_radiance",
            "optical_transmission",
            "D",
            "f",
            "w_x",
            "w_y",
            "qe",
            "other_irradiance",
            "dark_current",
            "expected",
        ),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                (1.0, np.array([1.62831853]), np.array([8.19714543e24])),
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                (
                    1.0,
                    np.array([1.62831853, 1.62831853]),
                    np.array([8.19714543e24, 8.19714543e24]),
                ),
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                2.0,
                (
                    2.0,
                    np.array([1.62831853, 1.62831853]),
                    np.array([8.19714543e24, 8.19714543e24]),
                ),
            ),
        ],
    )
    def test_signal_rate(
        self,
        wavelengths: np.ndarray,
        target_radiance: np.ndarray,
        optical_transmission: np.ndarray,
        D: float,  # noqa: N803
        f: float,
        w_x: float,
        w_y: float,
        qe: np.ndarray,
        other_irradiance: np.ndarray,
        dark_current: float,
        expected: tuple[float, np.ndarray, np.ndarray],
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.signal_rate(
            wavelengths,
            target_radiance,
            optical_transmission,
            D,
            f,
            w_x,
            w_y,
            qe,
            other_irradiance,
            dark_current,
        )
        assert np.isclose(output[0], expected[0])
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize(
        ("wavelengths", "cold_filter_temperature", "cold_filter_emissivity", "D", "f"),
        [
            (np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), 1.0, 1.0, 0.0, 1.0),
        ],
    )
    def test_cold_stop_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        cold_filter_temperature: float,
        cold_filter_emissivity: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            radiance.cold_stop_self_emission(
                wavelengths,
                cold_filter_temperature,
                cold_filter_emissivity,
                D,
                f,
            )

    @pytest.mark.parametrize(
        ("wavelengths", "cold_filter_temperature", "cold_filter_emissivity", "D", "f"),
        [
            (np.array([]), 0.0, 0.0, 1.0, 0.0),
            (np.array([]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_cold_stop_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        cold_filter_temperature: float,
        cold_filter_emissivity: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.cold_stop_self_emission(
            wavelengths,
            cold_filter_temperature,
            cold_filter_emissivity,
            D,
            f,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "cold_filter_temperature",
            "cold_filter_emissivity",
            "D",
            "f",
            "expected",
        ),
        [
            (np.array([1.0]), 0.0, 0.0, 1.0, 0.0, np.array([0.0])),
            (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([5.16399351e-15, 5.16399351e-15]),
            ),
        ],
    )
    def test_cold_stop_self_emission(
        self,
        wavelengths: np.ndarray,
        cold_filter_temperature: float,
        cold_filter_emissivity: float,
        D: float,  # noqa: N803
        f: float,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.cold_stop_self_emission(
            wavelengths,
            cold_filter_temperature,
            cold_filter_emissivity,
            D,
            f,
        )
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "optics_temperature",
            "optics_emissivity",
            "cold_filter_transmission",
            "D",
            "f",
        ),
        [
            (np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0),
            (np.array([]), 1.0, 1.0, 1.0, 0.0, 1.0),
        ],
    )
    def test_optics_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        optics_temperature: float,
        optics_emissivity: float,
        cold_filter_transmission: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            radiance.optics_self_emission(
                wavelengths,
                optics_temperature,
                optics_emissivity,
                cold_filter_transmission,
                D,
                f,
            )

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "optics_temperature",
            "optics_emissivity",
            "cold_filter_transmission",
            "D",
            "f",
        ),
        [
            (np.array([]), 0.0, 0.0, 0.0, 1.0, 0.0),
            (np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_optics_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        optics_temperature: float,
        optics_emissivity: float,
        cold_filter_transmission: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.optics_self_emission(
            wavelengths,
            optics_temperature,
            optics_emissivity,
            cold_filter_transmission,
            D,
            f,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "optics_temperature",
            "optics_emissivity",
            "cold_filter_transmission",
            "D",
            "f",
            "expected",
        ),
        [
            (np.array([1.0]), 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0])),
            (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.array([5.16399351e-15, 5.16399351e-15]),
            ),
        ],
    )
    def test_optics_self_emission(
        self,
        wavelengths: np.ndarray,
        optics_temperature: float,
        optics_emissivity: float,
        cold_filter_transmission: float,
        D: float,  # noqa: N803
        f: float,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.optics_self_emission(
            wavelengths,
            optics_temperature,
            optics_emissivity,
            cold_filter_transmission,
            D,
            f,
        )
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("wavelengths", "cold_shield_temperature", "D", "f"),
        [
            (np.array([]), 0.0, 0.0, 0.0),
            (np.array([]), 1.0, 0.0, 1.0),
        ],
    )
    def test_cold_shield_self_emission_zero_division(
        self,
        wavelengths: np.ndarray,
        cold_shield_temperature: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            radiance.cold_shield_self_emission(
                wavelengths,
                cold_shield_temperature,
                D,
                f,
            )

    @pytest.mark.parametrize(
        ("wavelengths", "cold_shield_temperature", "D", "f"),
        [
            (np.array([]), 0.0, 1.0, 0.0),
            (np.array([]), 1.0, 1.0, 1.0),
        ],
    )
    def test_cold_shield_self_emission_empty_array(
        self,
        wavelengths: np.ndarray,
        cold_shield_temperature: float,
        D: float,  # noqa: N803
        f: float,
    ) -> None:
        """Cover cases where output is an empty array."""
        output = radiance.cold_shield_self_emission(
            wavelengths,
            cold_shield_temperature,
            D,
            f,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        ("wavelengths", "cold_shield_temperature", "D", "f", "expected"),
        [
            (np.array([1.0]), 1.0, 1.0, 1.0, np.array([5.16399351e-15])),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                np.array([5.16399351e-15, 5.16399351e-15]),
            ),
        ],
    )
    def test_cold_shield_self_emission(
        self,
        wavelengths: np.ndarray,
        cold_shield_temperature: float,
        D: float,  # noqa: N803
        f: float,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.cold_shield_self_emission(
            wavelengths,
            cold_shield_temperature,
            D,
            f,
        )
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("atm", "reflectance", "temperature"),
        [
            (np.array([]), 0.0, 0.0),
            (np.array([]), 1.0, 1.0),
            (np.array([1.0]), 1.0, 1.0),
        ],
    )
    def test_total_radiance_index_error(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            radiance.total_radiance(atm, reflectance, temperature)

    @pytest.mark.parametrize(
        ("atm", "reflectance", "temperature", "expected"),
        [
            (np.ones(shape=(6, 6)), 0.0, 0.0, np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])),
            (
                np.ones(shape=(6, 6)),
                1.0,
                1.0,
                np.array(
                    [
                        8.66666667,
                        8.66666667,
                        8.66666667,
                        8.66666667,
                        8.66666667,
                        8.66666667,
                    ],
                ),
            ),
        ],
    )
    def test_total_radiance(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.total_radiance(atm, reflectance, temperature)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("atm", "reflectance", "temperature", "expected"),
        [
            (
                utils.load_database_atmosphere(1000.0, 0.0, 1),
                0.0,
                0.0,
                np.load("./tests/radiance/data/total_radiance_atm_zero.npy"),
            ),
            (
                utils.load_database_atmosphere(1000.0, 0.0, 1),
                1.0,
                1.0,
                np.load("./tests/radiance/data/total_radiance_atm_one.npy"),
            ),
        ],
    )
    def test_total_radiance_atm(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        expected: np.ndarray,
    ) -> None:
        """Cover cases with normal inputs, atm input, and expected outputs."""
        output = radiance.total_radiance(atm, reflectance, temperature)
        assert np.isclose(output, expected).all()

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp"),
        [
            (np.array([]), Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 0.0, 0),
            (np.array([]), Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])), 1.0, 1),
            (
                np.array([1.0]),
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                1.0,
                1,
            ),
        ],
    )
    def test_reflectance_to_photoelectrons_index_error(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        int_time: float,
        target_temp: int,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            radiance.reflectance_to_photoelectrons(atm, sensor, int_time, target_temp)

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp", "expected"),
        [
            (
                np.ones(shape=(6, 6)),
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                0.0,
                0,
                (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6))),
            ),
            (
                np.ones(shape=(6, 6)),
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                1.0,
                300,
                (np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(shape=(2, 6))),
            ),
        ],
    )
    def test_reflectance_to_photoelectrons(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        int_time: float,
        target_temp: int,
        expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.reflectance_to_photoelectrons(
            atm,
            sensor,
            int_time,
            target_temp,
        )
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp", "expected"),
        [
            (
                utils.load_database_atmosphere(1000.0, 0.0, 1),
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                1.0,
                300,
                (
                    np.linspace(0.0, 1.0, 100),
                    np.load(
                        "./tests/radiance/data/reflectance_to_photoelectrons_pe.npy",
                    ),
                    np.load(
                        "./tests/radiance/data/reflectance_to_photoelectrons_spectral_weights.npy",
                    ),
                ),
            ),
        ],
    )
    def test_reflectance_to_photoelectrons_atm(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        int_time: float,
        target_temp: int,
        expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Cover cases with normal inputs, atm input, and expected outputs."""
        output = radiance.reflectance_to_photoelectrons(
            atm,
            sensor,
            int_time,
            target_temp,
        )
        assert np.isclose(output[0], expected[0]).all()
        assert np.isclose(output[1], expected[1]).all()
        assert np.isclose(output[2], expected[2]).all()

    @pytest.mark.parametrize(
        (
            "L",
            "L_s",
            "t_opt",
            "e_opt",
            "lambda0",
            "d_lambda",
            "optics_temperature",
            "D",
            "f",
            "expectation",
        ),
        [
            (
                np.array([]),
                0.0,
                0.0,
                0.0,
                np.array([]),
                0.0,
                0.0,
                0.0,
                0.0,
                pytest.raises(ZeroDivisionError),
            ),
            (
                np.array([]),
                1.0,
                1.0,
                1.0,
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                does_not_raise(),
            ),
            (
                np.array([1.0, 2.5, 3.0]),
                1.0,
                1.0,
                1.0,
                np.array([1.0, 2.5, 3.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                does_not_raise(),
            ),
            (
                np.array([1.0, 2.5, 3.0]),
                0.0001,
                0.84,
                0.16,
                np.array([1.0, 2.5, 3.0]),
                2.5,
                70.0,
                4.0,
                275e-3,
                does_not_raise(),
            ),
            (
                np.ones((10, 10)),
                1.0,
                1.0,
                1.0,
                np.ones((10, 10)),
                1.0,
                1.0,
                1.0,
                1.0,
                does_not_raise(),
            ),
        ],
    )
    def test_focal_plane_integrated_irradiance(
        self,
        snapshot: SnapshotAssertion,
        L: np.ndarray,  # noqa: N803
        L_s: float,  # noqa: N803
        t_opt: float,
        e_opt: float,
        lambda0: np.ndarray,
        d_lambda: float,
        optics_temperature: float,
        D: float,  # noqa: N803
        f: float,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            output = radiance.focal_plane_integrated_irradiance(
                L,
                L_s,
                t_opt,
                e_opt,
                lambda0,
                d_lambda,
                optics_temperature,
                D,
                f,
            )
            assert np.all(output == snapshot)

    @pytest.mark.parametrize(
        ("total_photoelectrons", "max_fill"),
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 1.5),
        ],
    )
    def test_check_well_fill(
        self,
        snapshot: SnapshotAssertion,
        total_photoelectrons: float,
        max_fill: float,
    ) -> None:
        output = radiance.check_well_fill(total_photoelectrons, max_fill)
        assert output == snapshot

    @pytest.mark.parametrize(
        ("sensor", "radiance_wavelengths", "target_radiance", "background_radiance"),
        [
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([]),
                np.array([]),
                np.array([]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([1.0]),
                np.array([]),
                np.array([]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([]),
                np.array([1.0]),
                np.array([]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([]),
                np.array([]),
                np.array([1.0]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.array([1.0, 2.5, 3.0]),
                np.array([1.0, 2.5, 3.0]),
                np.array([1.0, 2.5, 3.0]),
            ),
            (
                Sensor("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                np.ones((5, 5)),
                np.ones((5, 5)),
                np.ones((5, 5)),
            ),
        ],
    )
    def test_photon_detector_snr(
        self,
        snapshot: SnapshotAssertion,
        sensor: Sensor,
        radiance_wavelengths: np.ndarray,
        target_radiance: np.ndarray,
        background_radiance: np.ndarray,
    ) -> None:
        output = radiance.photon_detector_SNR(
            sensor,
            radiance_wavelengths,
            target_radiance,
            background_radiance,
        )
        assert output == snapshot

    @pytest.mark.parametrize(
        ("atm", "is_emissive", "expectation"),
        [
            (np.array([]), 0, pytest.raises(IndexError)),
            (np.ones((6, 6)), 0, does_not_raise()),
            (np.ones((6, 6)), 1, does_not_raise()),
            (utils.load_database_atmosphere(1000.0, 0.0, 1), 1, does_not_raise()),
        ],
    )
    def test_giqe_radiance(
        self,
        snapshot: SnapshotAssertion,
        atm: np.ndarray,
        is_emissive: int,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            output = radiance.giqe_radiance(atm, is_emissive)
            assert output == snapshot
