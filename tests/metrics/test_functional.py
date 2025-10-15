from __future__ import annotations

import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm.metrics import functional
from pybsm.simulation import Scenario, Sensor


class TestFunctional:
    @pytest.mark.parametrize(
        ("rer", "gsd", "eho", "ng", "snr", "expectation"),
        [
            (1.0, 1.0, 1.0, 2.0, 0.0, pytest.raises(ZeroDivisionError)),
            (2.0, 0.0, 2.0, 1.0, 1.0, pytest.raises(ZeroDivisionError)),
            (0.0, 2.0, 0.0, 0.0, 2.0, does_not_raise()),
        ],
    )
    def test_giqe3_error(
        self,
        rer: float,
        gsd: float,
        eho: float,
        ng: float,
        snr: float,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            functional.giqe3(
                rer=rer,
                gsd=gsd,
                eho=eho,
                ng=ng,
                snr=snr,
            )

    @pytest.mark.parametrize(
        ("rer", "gsd", "eho", "ng", "snr"),
        [
            (1.0, 1.0, 1.0, 2.0, 2.0),
            (2.0, 2.0, 2.0, 1.0, 1.0),
        ],
    )
    def test_giqe3(
        self,
        rer: float,
        gsd: float,
        eho: float,
        ng: float,
        snr: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.giqe3(
            rer=rer,
            gsd=gsd,
            eho=eho,
            ng=ng,
            snr=snr,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("rer", "gsd", "eho", "ng", "snr", "elev_angle", "expectation"),
        [
            (1.0, 1.0, 1.0, 2.0, 0.0, 10.0, pytest.raises(ZeroDivisionError)),
            (0.0, 2.0, 0.0, 0.0, 2.0, 90.0, does_not_raise()),
        ],
    )
    def test_giqe4_error(
        self,
        rer: float,
        gsd: float,
        eho: float,
        ng: float,
        snr: float,
        elev_angle: float,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            functional.giqe4(
                rer=rer,
                gsd=gsd,
                eho=eho,
                ng=ng,
                snr=snr,
                elev_angle=elev_angle,
            )

    @pytest.mark.parametrize(
        ("rer", "gsd", "eho", "ng", "snr", "elev_angle"),
        [
            (1.0, 1.0, 1.0, 2.0, 2.0, 0.0),
            (2.0, 2.0, 2.0, 1.0, 1.0, np.pi / 2),
        ],
    )
    def test_giqe4(
        self,
        rer: float,
        gsd: float,
        eho: float,
        ng: float,
        snr: float,
        elev_angle: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.giqe4(
            rer=rer,
            gsd=gsd,
            eho=eho,
            ng=ng,
            snr=snr,
            elev_angle=elev_angle,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("rer_1", "rer_2", "gsd", "snr", "elev_angle", "expectation"),
        [(1.0, 1.0, 1.0, 0.0, 20.0, pytest.raises(ZeroDivisionError)), (0.0, 2.0, 0.0, 2.0, 90.0, does_not_raise())],
    )
    def test_giqe5_error(
        self,
        rer_1: float,
        rer_2: float,
        gsd: float,
        snr: float,
        elev_angle: float,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            functional.giqe5(
                rer_1=rer_1,
                rer_2=rer_2,
                gsd=gsd,
                snr=snr,
                elev_angle=elev_angle,
            )

    @pytest.mark.parametrize(
        ("rer_1", "rer_2", "gsd", "snr", "elev_angle"),
        [
            (1.0, 1.0, 1.0, 2.0, 20.0),
            (2.0, 2.0, 2.0, 1.0, 90.0),
            (0.35, 0.35, 0.52832, 50, np.pi / 2),  # From NGS GIQE5 paper
        ],
    )
    def test_giqe5(
        self,
        rer_1: float,
        rer_2: float,
        gsd: float,
        snr: float,
        elev_angle: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.giqe5(
            rer_1=rer_1,
            rer_2=rer_2,
            gsd=gsd,
            snr=snr,
            elev_angle=elev_angle,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("pixel_pos", "mtf_slice", "df", "ifov"),
        [
            (1.0, np.array([1.0, 2.0]), 5.0, np.pi / 2),
            (5.0, np.array([1.0, 1.0]), 10.0, np.pi / 2),
            (2.0, np.array([0.5, 0.5]), 1.0, np.pi / 4),
        ],
    )
    def test_edge_response(
        self,
        pixel_pos: float,
        mtf_slice: np.ndarray,
        df: float,
        ifov: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.edge_response(
            pixel_pos=pixel_pos,
            mtf_slice=mtf_slice,
            df=df,
            ifov=ifov,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("mtf_slice", "df", "ifov"),
        [
            (np.array([1.0, 2.0]), 5.0, np.pi / 2),
            (np.array([1.0, 1.0]), 10.0, 0),
            (np.array([0.5, 0.5]), 0.0, np.pi / 4),
        ],
    )
    def test_relative_edge_response(
        self,
        mtf_slice: np.ndarray,
        df: float,
        ifov: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.relative_edge_response(
            mtf_slice=mtf_slice,
            df=df,
            ifov=ifov,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("mtf_slice", "df", "ifov"),
        [
            (np.array([1.0, 2.0]), 5.0, np.pi / 2),
            (np.array([1.0, 1.0]), 10.0, 0),
            (np.array([0.5, 0.5]), 0.0, np.pi / 4),
        ],
    )
    def test_edge_height_overshoot(
        self,
        mtf_slice: np.ndarray,
        df: float,
        ifov: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.edge_height_overshoot(
            mtf_slice=mtf_slice,
            df=df,
            ifov=ifov,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("mtf", "df", "ifov_x", "ifov_y"),
        [
            (np.array([[1.0, 2.0], [1.0, 2.0]]), 5.0, np.pi / 2, np.pi / 2),
            (np.array([[1.0, 0.0], [1.0, 0.0]]), 10.0, 0, 0),
            (np.array([[0.0, 1.0], [0.0, 1.0]]), 1.0, 0, np.pi / 4),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), 0.0, np.pi / 4, 0),
        ],
    )
    def test_giqe5_RER(  # noqa: N802
        self,
        mtf: np.ndarray,
        df: float,
        ifov_x: float,
        ifov_y: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.giqe5_RER(
            mtf=mtf,
            df=df,
            ifov_x=ifov_x,
            ifov_y=ifov_y,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("mtf", "df", "ifov_x", "ifov_y"),
        [
            (np.array([[1.0, 2.0], [1.0, 2.0]]), 5.0, np.pi / 2, np.pi / 2),
            (np.array([[1.0, 0.0], [1.0, 0.0]]), 10.0, 0, 0),
            (np.array([[0.0, 1.0], [0.0, 1.0]]), 1.0, 0, np.pi / 4),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), 0.0, np.pi / 4, 0),
        ],
    )
    def test_giqe_edge_terms(
        self,
        mtf: np.ndarray,
        df: float,
        ifov_x: float,
        ifov_y: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.giqe_edge_terms(
            mtf=mtf,
            df=df,
            ifov_x=ifov_x,
            ifov_y=ifov_y,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("mtf_slice", "df", "snr", "ifov", "slant_range", "expectation"),
        [
            (np.array([1.0, 2.0]), 1.0, 0.0, np.pi / 2, 10.0, pytest.raises(ZeroDivisionError)),
            (np.array([0.5, 0.5]), 0.0, 5.0, np.pi / 4, 0.0, does_not_raise()),
        ],
    )
    def test_ground_resolved_distance_error(
        self,
        mtf_slice: np.ndarray,
        df: float,
        snr: float,
        ifov: float,
        slant_range: float,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            functional.ground_resolved_distance(
                mtf_slice=mtf_slice,
                df=df,
                snr=snr,
                ifov=ifov,
                slant_range=slant_range,
            )

    @pytest.mark.parametrize(
        ("mtf_slice", "df", "snr", "ifov", "slant_range"),
        [
            (np.array([1.0, 2.0]), 1.0, 5.0, np.pi / 2, 0.0),
            (np.array([1.0, 1.0]), 0.0, 10.0, 0, 1.0),
            (np.array([0.5, 0.5]), 0.0, 1.0, np.pi / 4, 10.0),
        ],
    )
    def test_ground_resolved_distance(
        self,
        mtf_slice: np.ndarray,
        df: float,
        snr: float,
        ifov: float,
        slant_range: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.ground_resolved_distance(
            mtf_slice=mtf_slice,
            df=df,
            snr=snr,
            ifov=ifov,
            slant_range=slant_range,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("sensor", "scenario", "interp"),
        [
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([2.91, 3.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                True,
            ),
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                False,
            ),
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                None,
            ),
        ],
    )
    def test_niirs(
        self,
        sensor: Sensor,
        scenario: Scenario,
        interp: bool | None,
        snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.niirs(sensor=sensor, scenario=scenario, interp=interp)
        assert snapshot == output

    @pytest.mark.parametrize(
        ("sensor", "scenario", "interp"),
        [
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([2.91, 3.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                True,
            ),
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                False,
            ),
            (
                Sensor(
                    name="test_sensor",
                    D=275e-3,
                    f=4,
                    p_x=0.008e-3,
                    opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
                ),
                Scenario(
                    name="test_scenario",
                    ihaze=1,
                    altitude=9000,
                    ground_range=0.0,
                ),
                None,
            ),
        ],
    )
    def test_niirs5(
        self,
        sensor: Sensor,
        scenario: Scenario,
        interp: bool | None,
        snapshot: SnapshotAssertion,
    ) -> None:
        output = functional.niirs5(sensor=sensor, scenario=scenario, interp=interp)
        assert snapshot == output

    @mock.patch("pybsm.metrics.functional.plt")
    def test_plot_common_MTFs(self, mock_plt: mock.MagicMock, snapshot: SnapshotAssertion) -> None:  # noqa: N802
        sensor = Sensor(
            name="test_sensor",
            D=275e-3,
            f=4,
            p_x=0.008e-3,
            opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
        )
        scenario = Scenario(
            name="test_scenario",
            ihaze=1,
            altitude=9000,
            ground_range=0.0,
        )
        metric = functional.niirs(sensor=sensor, scenario=scenario)
        functional.plot_common_MTFs(metrics=metric, orientation_angle=np.pi / 2)
        assert mock_plt.plot.call_count == 2
        call_args, _ = mock_plt.annotate.call_args
        assert call_args == snapshot
        # Verify xlabel was called with correct parameters
        mock_plt.xlabel.assert_called_once_with("spatial frequency (cycles/mm)")

        # Verify ylabel was called with correct parameters
        mock_plt.ylabel.assert_called_once_with("MTF")

        mock_plt.legend.assert_called_once_with(
            [
                "aperture",
                "turbulence",
                "detector",
                "jitter",
                "drift",
                "wavefront",
                "image processing",
                "system",
            ],
        )

        # Verify annotate was called once
        mock_plt.annotate.assert_called_once()

    @mock.patch("pybsm.metrics.functional.plt")
    def test_plot_noise_terms(self, mock_plt: mock.MagicMock) -> None:
        mock_fig = mock.MagicMock()
        mock_ax = mock.MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        sensor = Sensor(
            name="test_sensor",
            D=275e-3,
            f=4,
            p_x=0.008e-3,
            opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
        )
        scenario = Scenario(
            name="test_scenario",
            ihaze=1,
            altitude=9000,
            ground_range=0.0,
        )
        metric = functional.niirs(sensor=sensor, scenario=scenario)
        functional.plot_noise_terms(metrics=metric, max_val=1)
        assert mock_plt.subplots.call_count == 1
        mock_plt.title.assert_called_once()

        mock_plt.ylim.assert_called_once_with([0, 1])

        mock_plt.tight_layout.assert_called_once()
