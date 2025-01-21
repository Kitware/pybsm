import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import utils
from tests.test_utils import CustomFloatSnapshotExtension


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


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
            utils.load_database_atmosphere_no_interp(altitude, ground_range, ihaze)

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
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test load_database_atmosphere_no_interp with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere_no_interp(altitude, ground_range, ihaze)
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range"),
        [
            (-1, 1000.0, 0.0),
            (0, 1000.0, 0.0),
            (1, 1.0, 0.0),
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
            utils.load_database_atmosphere(altitude, ground_range, ihaze)

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
        ],
    )
    def test_load_database_atmosphere(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test load_database_atmosphere with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere(altitude, ground_range, ihaze)
        snapshot_custom.assert_match(output)
