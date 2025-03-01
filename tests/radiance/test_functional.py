from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import radiance, utils
from pybsm.simulation import Sensor
from tests.test_utils import CustomFloatSnapshotExtension


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


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
        output = radiance.photon_detection_rate(
            E=E,
            w_x=w_x,
            w_y=w_y,
            wavelengths=wavelengths,
            qe=qe,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        ("E", "w_x", "w_y", "wavelengths", "qe"),
        [
            (
                np.array([1.0]),
                1.0,
                1.0,
                np.array([1.0]),
                np.array([1.0]),
            ),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.photon_detection_rate(
            E=E,
            w_x=w_x,
            w_y=w_y,
            wavelengths=wavelengths,
            qe=qe,
        )
        snapshot_custom.assert_match(output)

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
            radiance.at_focal_plane_irradiance(D=D, f=f, L=L)

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
        output = radiance.at_focal_plane_irradiance(D=D, f=f, L=L)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("D", "f", "L"),
        [
            (1.0, 0.0, np.array([1.0])),
            (1.0, 1.0, np.array([1.0])),
            (1.0, 1.0, np.array([1.0, 1.0])),
            (1.0, 1.0, np.array([1.0, 2.0])),
        ],
    )
    def test_at_focal_plane_irradiance(
        self,
        D: float,  # noqa: N803
        f: float,
        L: np.ndarray,  # noqa: N803
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.at_focal_plane_irradiance(D=D, f=f, L=L)
        snapshot_custom.assert_match(output)

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
        output = radiance.blackbody_radiance(lambda0=lambda0, T=T)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("lambda0", "T"),
        [
            (np.array([1.0]), 1.0),
            (np.array([1.0, 1.0]), 1.0),
        ],
    )
    def test_blackbody_radiance(
        self,
        lambda0: np.ndarray,
        T: float,  # noqa: N803
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.blackbody_radiance(lambda0=lambda0, T=T)
        snapshot_custom.assert_match(output)

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
            radiance.resample_by_wavelength(
                wavelengths=wavelengths,
                values=values,
                new_wavelengths=new_wavelengths,
            )

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
        output = radiance.resample_by_wavelength(
            wavelengths=wavelengths,
            values=values,
            new_wavelengths=new_wavelengths,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        ("wavelengths", "values", "new_wavelengths"),
        [
            (np.array([1.0]), np.array([1.0]), np.array([1.0])),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 3.0]),
                np.array([1.0, 1.5, 2.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0, 3.0]),
            ),
        ],
    )
    def test_resample_by_wavelength(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        new_wavelengths: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.resample_by_wavelength(
            wavelengths=wavelengths,
            values=values,
            new_wavelengths=new_wavelengths,
        )
        snapshot_custom.assert_match(output)

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
                wavelengths=wavelengths,
                target_radiance=target_radiance,
                optical_transmission=optical_transmission,
                D=D,
                f=f,
                w_x=w_x,
                w_y=w_y,
                qe=qe,
                other_irradiance=other_irradiance,
                dark_current=dark_current,
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases where both output arrays are empty."""
        output = radiance.signal_rate(
            wavelengths=wavelengths,
            target_radiance=target_radiance,
            optical_transmission=optical_transmission,
            D=D,
            f=f,
            w_x=w_x,
            w_y=w_y,
            qe=qe,
            other_irradiance=other_irradiance,
            dark_current=dark_current,
        )
        snapshot_custom.assert_match(output[0])
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases where second output array is empty."""
        output = radiance.signal_rate(
            wavelengths=wavelengths,
            target_radiance=target_radiance,
            optical_transmission=optical_transmission,
            D=D,
            f=f,
            w_x=w_x,
            w_y=w_y,
            qe=qe,
            other_irradiance=other_irradiance,
            dark_current=dark_current,
        )
        snapshot_custom.assert_match(output[1:])
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.signal_rate(
            wavelengths=wavelengths,
            target_radiance=target_radiance,
            optical_transmission=optical_transmission,
            D=D,
            f=f,
            w_x=w_x,
            w_y=w_y,
            qe=qe,
            other_irradiance=other_irradiance,
            dark_current=dark_current,
        )
        snapshot_custom.assert_match(output)

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
                wavelengths=wavelengths,
                cold_filter_temperature=cold_filter_temperature,
                cold_filter_emissivity=cold_filter_emissivity,
                D=D,
                f=f,
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
            wavelengths=wavelengths,
            cold_filter_temperature=cold_filter_temperature,
            cold_filter_emissivity=cold_filter_emissivity,
            D=D,
            f=f,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        (
            "wavelengths",
            "cold_filter_temperature",
            "cold_filter_emissivity",
            "D",
            "f",
        ),
        [
            (np.array([1.0]), 0.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), 1.0, 1.0, 1.0, 1.0),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.cold_stop_self_emission(
            wavelengths=wavelengths,
            cold_filter_temperature=cold_filter_temperature,
            cold_filter_emissivity=cold_filter_emissivity,
            D=D,
            f=f,
        )
        snapshot_custom.assert_match(output)

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
                wavelengths=wavelengths,
                optics_temperature=optics_temperature,
                optics_emissivity=optics_emissivity,
                cold_filter_transmission=cold_filter_transmission,
                D=D,
                f=f,
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
            wavelengths=wavelengths,
            optics_temperature=optics_temperature,
            optics_emissivity=optics_emissivity,
            cold_filter_transmission=cold_filter_transmission,
            D=D,
            f=f,
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
        ),
        [
            (np.array([1.0]), 0.0, 0.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.optics_self_emission(
            wavelengths=wavelengths,
            optics_temperature=optics_temperature,
            optics_emissivity=optics_emissivity,
            cold_filter_transmission=cold_filter_transmission,
            D=D,
            f=f,
        )
        snapshot_custom.assert_match(output)

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
                wavelengths=wavelengths,
                cold_shield_temperature=cold_shield_temperature,
                D=D,
                f=f,
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
            wavelengths=wavelengths,
            cold_shield_temperature=cold_shield_temperature,
            D=D,
            f=f,
        )
        assert output.size == 0

    @pytest.mark.parametrize(
        ("wavelengths", "cold_shield_temperature", "D", "f"),
        [
            (np.array([1.0]), 1.0, 1.0, 1.0),
            (
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_cold_shield_self_emission(
        self,
        wavelengths: np.ndarray,
        cold_shield_temperature: float,
        D: float,  # noqa: N803
        f: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.cold_shield_self_emission(
            wavelengths=wavelengths,
            cold_shield_temperature=cold_shield_temperature,
            D=D,
            f=f,
        )
        snapshot_custom.assert_match(output)

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
            radiance.total_radiance(
                atm=atm,
                reflectance=reflectance,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        ("atm", "reflectance", "temperature"),
        [
            (np.ones(shape=(6, 6)), 0.0, 0.0),
            (
                np.ones(shape=(6, 6)),
                1.0,
                1.0,
            ),
        ],
    )
    def test_total_radiance(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.total_radiance(
            atm=atm,
            reflectance=reflectance,
            temperature=temperature,
        )
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("atm", "reflectance", "temperature"),
        [
            (
                utils.load_database_atmosphere(
                    altitude=1000.0,
                    ground_range=0.0,
                    ihaze=1,
                ),
                0.0,
                0.0,
            ),
            (
                utils.load_database_atmosphere(
                    altitude=1000.0,
                    ground_range=0.0,
                    ihaze=1,
                ),
                1.0,
                1.0,
            ),
        ],
    )
    def test_total_radiance_atm(
        self,
        atm: np.ndarray,
        reflectance: float,
        temperature: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs, atm input, and expected outputs."""
        output = radiance.total_radiance(
            atm=atm,
            reflectance=reflectance,
            temperature=temperature,
        )
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp"),
        [
            (
                np.array([]),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                0.0,
                0,
            ),
            (
                np.array([]),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                1.0,
                1,
            ),
            (
                np.array([1.0]),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
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
            radiance.reflectance_to_photoelectrons(
                atm=atm,
                sensor=sensor,
                int_time=int_time,
                target_temp=target_temp,
            )

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp"),
        [
            (
                np.ones(shape=(6, 6)),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                0.0,
                0,
            ),
            (
                np.ones(shape=(6, 6)),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                1.0,
                300,
            ),
        ],
    )
    def test_reflectance_to_photoelectrons(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        int_time: float,
        target_temp: int,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs and expected outputs."""
        output = radiance.reflectance_to_photoelectrons(
            atm=atm,
            sensor=sensor,
            int_time=int_time,
            target_temp=target_temp,
        )
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("atm", "sensor", "int_time", "target_temp"),
        [
            (
                utils.load_database_atmosphere(
                    altitude=1000.0,
                    ground_range=0.0,
                    ihaze=1,
                ),
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                1.0,
                300,
            ),
        ],
    )
    def test_reflectance_to_photoelectrons_atm(
        self,
        atm: np.ndarray,
        sensor: Sensor,
        int_time: float,
        target_temp: int,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Cover cases with normal inputs, atm input, and expected outputs."""
        output = radiance.reflectance_to_photoelectrons(
            atm=atm,
            sensor=sensor,
            int_time=int_time,
            target_temp=target_temp,
        )
        snapshot_custom.assert_match(output)

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
                L=L,
                L_s=L_s,
                t_opt=t_opt,
                e_opt=e_opt,
                lambda0=lambda0,
                d_lambda=d_lambda,
                optics_temperature=optics_temperature,
                D=D,
                f=f,
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
        output = radiance.check_well_fill(
            total_photoelectrons=total_photoelectrons,
            max_fill=max_fill,
        )
        assert output == snapshot

    @pytest.mark.parametrize(
        ("sensor", "radiance_wavelengths", "target_radiance", "background_radiance"),
        [
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([]),
                np.array([]),
                np.array([]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([1.0]),
                np.array([]),
                np.array([]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([]),
                np.array([1.0]),
                np.array([]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([]),
                np.array([]),
                np.array([1.0]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
                np.array([1.0, 2.5, 3.0]),
                np.array([1.0, 2.5, 3.0]),
                np.array([1.0, 2.5, 3.0]),
            ),
            (
                Sensor(
                    name="Test",
                    D=1.0,
                    f=1.0,
                    p_x=1.0,
                    opt_trans_wavelengths=np.array([0.0, 1.0]),
                ),
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
            sensor=sensor,
            radiance_wavelengths=radiance_wavelengths,
            target_radiance=target_radiance,
            background_radiance=background_radiance,
        )
        assert output == snapshot

    @pytest.mark.parametrize(
        ("atm", "is_emissive", "expectation"),
        [
            (np.array([]), 0, pytest.raises(IndexError)),
            (np.ones((6, 6)), 0, does_not_raise()),
            (np.ones((6, 6)), 1, does_not_raise()),
            (
                utils.load_database_atmosphere(
                    altitude=1000.0,
                    ground_range=0.0,
                    ihaze=1,
                ),
                1,
                does_not_raise(),
            ),
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
            output = radiance.giqe_radiance(atm=atm, is_emissive=is_emissive)
            assert output == snapshot
