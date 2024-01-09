import numpy as np
import pytest

from pybsm import utils
from typing import ContextManager
from contextlib import nullcontext as does_not_raise


class TestUtils:

    @pytest.mark.parametrize('ihaze, altitude, ground_range, expected, expectation', [
        (-1, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (0, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (1, 1000.0, 0.0, np.load('./tests/scenario/data/1_1000_0_atm.npy'), does_not_raise()),
        (1, 1.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (1, 300.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (1, 1000.0, 500.0, np.load('./tests/scenario/data/1_1000_500_atm.npy'), does_not_raise()),
        (1, 1000.0, 300.0, np.array([]), pytest.raises(IndexError)),
    ])
    def test_load_database_atmosphere_nointerp(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        with expectation:
            output = utils.loadDatabaseAtmosphere_nointerp(altitude, ground_range, ihaze)
            assert np.isclose(output, expected).all()

    @pytest.mark.parametrize('ihaze, altitude, ground_range, expected, expectation', [
        (-1, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (0, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (1, 1000.0, 0.0, np.load('./tests/scenario/data/1_1000_0_atm.npy'), does_not_raise()),
        (1, 1.0, 0.0, np.array([]), pytest.raises(IndexError)),
        (1, 300.0, 0.0, np.load('./tests/scenario/data/1_300_0_atm.npy'), does_not_raise()),
        (1, 1000.0, 500.0, np.load('./tests/scenario/data/1_1000_500_atm.npy'), does_not_raise()),
        (1, 1000.0, 300.0, np.load('./tests/scenario/data/1_1000_300_atm.npy'), does_not_raise()),
    ])
    def test_load_database_atmosphere(
        self,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        with expectation:
            output = utils.loadDatabaseAtmosphere(altitude, ground_range, ihaze)
            assert np.isclose(output, expected).all()
