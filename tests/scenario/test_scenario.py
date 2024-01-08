import numpy as np
import pytest

from pybsm.simulation import Scenario
from pybsm import utils
from typing import ContextManager, Dict
from contextlib import nullcontext as does_not_raise

class TestScenario:

    def check_scenario(self, scenario: Scenario, name: str, ihaze: int, altitude: float,
                       ground_range: float, aircraftSpeed: float = 0, targetReflectance: float = 0.15,
                       targetTemperature: float = 295, backgroundReflectance: float = 0.07,
                       backgroundTemperature: float = 293, haWindspeed: float = 21, cn2at1m: float = 1.7e-14) -> None:
        assert scenario.name == name
        assert scenario.ihaze == ihaze
        assert scenario.altitude == altitude
        assert scenario.ground_range == ground_range
        assert scenario.aircraftSpeed == aircraftSpeed
        assert scenario.targetReflectance == targetReflectance
        assert scenario.targetTemperature == targetTemperature
        assert scenario.backgroundReflectance == backgroundReflectance
        assert scenario.backgroundTemperature == backgroundTemperature
        assert scenario.haWindspeed == haWindspeed
        assert scenario.cn2at1m == cn2at1m

    @pytest.mark.parametrize('name, ihaze, altitude, ground_range, other_args, expectation', [
        ('', 0, 0.0, 0.0, {}, does_not_raise()),
        ('test', 1, 1.0, 1.0, {}, does_not_raise()),
        ('test', 1, 1.0, 1.0, {'aircraftSpeed': 1.0, 'targetReflectance': 1.0,
            'targetTemperature': 1.0, 'backgroundReflectance': 1.0, 'backgroundTemperature': 1.0,
            'haWindspeed': 1.0, 'cn2at1m': 1.0}, does_not_raise()),
    ])
    def test_initialization(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        other_args: Dict[str, float],
        expectation: ContextManager
    ) -> None:
        with expectation:
            scenario = Scenario(name, ihaze, altitude, ground_range, **other_args)
            self.check_scenario(scenario, name, ihaze, altitude, ground_range, **other_args)

    @pytest.mark.parametrize('original_ihaze, new_ihaze, expectation', [
        (0, 1, does_not_raise()),
    ])   
    def test_ihaze(
        self,
        original_ihaze: int,
        new_ihaze: int,
        expectation: ContextManager
    ) -> None:
        with expectation:
            scenario = Scenario('test', original_ihaze, 0.0, 0.0)
            self.check_scenario(scenario, 'test', original_ihaze, 0.0, 0.0)
            scenario.ihaze = new_ihaze
            assert scenario.ihaze != original_ihaze
            self.check_scenario(scenario, 'test', new_ihaze, 0.0, 0.0)

    @pytest.mark.parametrize('original_altitude, new_altitude, expectation', [
        (0.0, 1.0, does_not_raise()),
    ])   
    def test_altitude(
        self,
        original_altitude: float,
        new_altitude: float,
        expectation: ContextManager
    ) -> None:
        with expectation:
            scenario = Scenario('test', 0, original_altitude, 0.0)
            self.check_scenario(scenario, 'test', 0, original_altitude, 0.0)
            scenario.altitude = new_altitude
            assert scenario.altitude != original_altitude
            self.check_scenario(scenario, 'test', 0, new_altitude, 0.0)

    @pytest.mark.parametrize('original_ground_range, new_ground_range, expectation', [
        (0.0, 1.0, does_not_raise()),
    ])   
    def test_altitude(
        self,
        original_ground_range: float,
        new_ground_range: float,
        expectation: ContextManager
    ) -> None:
        with expectation:
            scenario = Scenario('test', 0, 0.0, original_ground_range)
            self.check_scenario(scenario, 'test', 0, 0.0, original_ground_range)
            scenario.ground_range = new_ground_range
            assert scenario.ground_range != original_ground_range
            self.check_scenario(scenario, 'test', 0, 0.0, new_ground_range)

    @pytest.mark.parametrize('name, ihaze, altitude, ground_range, expected, expectation', [
        ('test', -1, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        ('test', 0, 1000.0, 0.0, np.array([]), pytest.raises(IndexError)),
        ('test', 2, 1000.0, 0.0, utils.loadDatabaseAtmosphere(1000.0, 0.0, 2), does_not_raise()),
        ('test', 1, 1.0, 1.0, np.array([]), pytest.raises(IndexError)),
        ('test', 1, 1000.0, 0.0, utils.loadDatabaseAtmosphere(1000.0, 0.0, 1), does_not_raise()),
        ('test', 1, 1000.0, 5.0, np.array([]), pytest.raises(ValueError)),
        ('test', 1, 1000.0, 100.0, np.array([]), pytest.raises(ValueError)),
    ])
    def test_initialization(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray,
        expectation: ContextManager
    ) -> None:
        with expectation:
            scenario = Scenario(name, ihaze, altitude, ground_range)
            atm = scenario.atm
            assert np.isclose(atm, expected).all()

