import pytest
from syrupy.assertion import SnapshotAssertion

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
        self,
        h_target: float,
        h_sensor: float,
        slant_range: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            geospatial.nadir_angle(
                h_target=h_target,
                h_sensor=h_sensor,
                slant_range=slant_range,
            )

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range"),
        [
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_nadir_angle(
        self,
        h_target: float,
        h_sensor: float,
        slant_range: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Test nadir_angle with normal inputs and expected outputs."""
        output = geospatial.nadir_angle(
            h_target=h_target,
            h_sensor=h_sensor,
            slant_range=slant_range,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range"),
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
    )
    def test_altitude_along_zero_division(
        self,
        h_target: float,
        h_sensor: float,
        slant_range: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            geospatial.altitude_along_slant_path(
                h_target=h_target,
                h_sensor=h_sensor,
                slant_range=slant_range,
            )

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "slant_range"),
        [
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_altitude_along_slant_path(
        self,
        h_target: float,
        h_sensor: float,
        slant_range: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Test altitude_along_slant_path with normal inputs and expected outputs."""
        output = geospatial.altitude_along_slant_path(
            h_target=h_target,
            h_sensor=h_sensor,
            slant_range=slant_range,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("ifov", "slant_range"),
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 2.5),
        ],
    )
    def test_ground_sample_distance(self, fuzzy_snapshot: SnapshotAssertion, ifov: float, slant_range: float) -> None:
        """Test ground_sample_distance with normal inputs and expected outputs."""
        output = geospatial.ground_sample_distance(ifov=ifov, slant_range=slant_range)
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("h_target", "h_sensor", "ground_range"),
        [
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_curved_earth_slant_range(
        self,
        fuzzy_snapshot: SnapshotAssertion,
        h_target: float,
        h_sensor: float,
        ground_range: float,
    ) -> None:
        """Test curved_earth_slant_range with normal inputs and expected outputs."""
        output = geospatial.curved_earth_slant_range(
            h_target=h_target,
            h_sensor=h_sensor,
            ground_range=ground_range,
        )
        fuzzy_snapshot.assert_match(output)
