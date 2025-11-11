import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import utils


class TestUtils:
    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range"),
        [
            (-1, 1000.0, 0.0),
            (0, 1000.0, 0.0),
            (1, 1.0, 0.0),
            (1, 300.0, 0.0),
            (1, 1000.0, 300.0),
        ],
    )
    def test_load_database_atmosphere_no_interp_index_error(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            utils.load_database_atmosphere_no_interp(
                altitude=altitude,
                ground_range=ground_range,
                ihaze=ihaze,
            )

    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range"),
        [
            (1, 1000.0, 0.0),
            (1, 1000.0, 500.0),
        ],
    )
    def test_load_database_atmosphere_no_interp(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Test load_database_atmosphere_no_interp with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere_no_interp(
            altitude=altitude,
            ground_range=ground_range,
            ihaze=ihaze,
        )
        fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range"),
        [
            (-1, 1000.0, 0.0),
            (0, 1000.0, 0.0),
            (1, 1.0, 0.0),
            (1, 1000.0, -1.0),
            (1, 1000.0, 302e3),
        ],
    )
    def test_load_database_atmosphere_index_error(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            utils.load_database_atmosphere(
                altitude=altitude,
                ground_range=ground_range,
                ihaze=ihaze,
            )

    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range"),
        [
            (1, 1000.0, 0.0),
            (
                1,
                300.0,
                0.0,
            ),
            (1, 1000.0, 500.0),
            (1, 1000.0, 300.0),
            (1, 25000.0, 300.0),
        ],
    )
    def test_load_database_atmosphere(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Test load_database_atmosphere with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere(
            altitude=altitude,
            ground_range=ground_range,
            ihaze=ihaze,
        )
        fuzzy_snapshot.assert_match(output)
