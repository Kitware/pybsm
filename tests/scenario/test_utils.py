import numpy as np
import pytest

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
            utils.load_database_atmosphere_no_interp(altitude, ground_range, ihaze)

    @pytest.mark.parametrize(
        ("ihaze", "altitude", "ground_range", "expected"),
        [
            (1, 1000.0, 0.0, np.load("./tests/scenario/data/1_1000_0_atm.npy")),
            (1, 1000.0, 500.0, np.load("./tests/scenario/data/1_1000_500_atm.npy")),
        ],
    )
    def test_load_database_atmosphere_no_interp(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray,
    ) -> None:
        """Test load_database_atmosphere_no_interp with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere_no_interp(altitude, ground_range, ihaze)
        assert np.isclose(output, expected).all()

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
        ("ihaze", "altitude", "ground_range", "expected"),
        [
            (1, 1000.0, 0.0, np.load("./tests/scenario/data/1_1000_0_atm.npy")),
            (1, 300.0, 0.0, np.load("./tests/scenario/data/1_300_0_atm.npy")),
            (1, 1000.0, 500.0, np.load("./tests/scenario/data/1_1000_500_atm.npy")),
            (1, 1000.0, 300.0, np.load("./tests/scenario/data/1_1000_300_atm.npy")),
        ],
    )
    def test_load_database_atmosphere(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray,
    ) -> None:
        """Test load_database_atmosphere with normal inputs and expected outputs."""
        output = utils.load_database_atmosphere(altitude, ground_range, ihaze)
        assert np.isclose(output, expected).all()
