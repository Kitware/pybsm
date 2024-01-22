import numpy as np
import pytest

from pybsm.simulation import Scenario
from pybsm import utils
from typing import Dict


class TestScenario:

    def check_scenario(
        self,
        scenario: Scenario,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        aircraftSpeed: float = 0,
        targetReflectance: float = 0.15,
        targetTemperature: float = 295,
        backgroundReflectance: float = 0.07,
        backgroundTemperature: float = 293,
        haWindspeed: float = 21,
        cn2at1m: float = 1.7e-14
    ) -> None:
        """
        Check if created scenario matches expected parameters
        """
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

    @pytest.mark.parametrize('name, ihaze, altitude, ground_range, other_args', [
        ('', 0, 0.0, 0.0, {}),
        ('test', 1, 1.0, 1.0, {}),
        ('test', 1, 1.0, 1.0, {'aircraftSpeed': 1.0, 'targetReflectance': 1.0,
                               'targetTemperature': 1.0, 'backgroundReflectance': 1.0, 'backgroundTemperature': 1.0,
                               'haWindspeed': 1.0, 'cn2at1m': 1.0}),
    ])
    def test_initialization(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        other_args: Dict[str, float]
    ) -> None:
        """
        Test initialization with and without default parameters
        """
        scenario = Scenario(name, ihaze, altitude, ground_range, **other_args)
        self.check_scenario(scenario, name, ihaze, altitude, ground_range, **other_args)

    @pytest.mark.parametrize('original_ihaze, new_ihaze', [
        (0, 1),
    ])
    def test_ihaze(
        self,
        original_ihaze: int,
        new_ihaze: int
    ) -> None:
        """
<<<<<<< HEAD
        Test that setting the ihaze attribute appropriately updates the internal value as
=======
         Test that setting the ihaze attribute appropriately updates the internal value as
>>>>>>> Add descriptions and test protected objects
        well as clear the internal atm attribute.
        """
        scenario = Scenario('test', original_ihaze, 0.0, 0.0)
        self.check_scenario(scenario, 'test', original_ihaze, 0.0, 0.0)
        scenario._atm = np.array([0])
        scenario.ihaze = new_ihaze
        assert scenario._atm is None
        assert scenario._ihaze != original_ihaze
        assert scenario.ihaze != original_ihaze
        self.check_scenario(scenario, 'test', new_ihaze, 0.0, 0.0)

    @pytest.mark.parametrize('original_altitude, new_altitude', [
        (0.0, 1.0),
    ])
    def test_altitude(
        self,
        original_altitude: float,
        new_altitude: float
    ) -> None:
        """
<<<<<<< HEAD
        Test that setting the altitude attribute appropriately updates the internal value as
=======
         Test that setting the altitude attribute appropriately updates the internal value as
>>>>>>> Add descriptions and test protected objects
        well as clear the internal atm attribute.
        """
        scenario = Scenario('test', 0, original_altitude, 0.0)
        self.check_scenario(scenario, 'test', 0, original_altitude, 0.0)
        scenario._atm = np.array([0])
        scenario.altitude = new_altitude
        assert scenario._atm is None
        assert scenario._altitude != original_altitude
        assert scenario.altitude != original_altitude
        self.check_scenario(scenario, 'test', 0, new_altitude, 0.0)

    @pytest.mark.parametrize('original_ground_range, new_ground_range', [
        (0.0, 1.0),
    ])
    def test_ground_range(
        self,
        original_ground_range: float,
        new_ground_range: float,
    ) -> None:
        """
<<<<<<< HEAD
        Test that setting the ground_range attribute appropriately updates the internal value as
=======
         Test that setting the ground_range attribute appropriately updates the internal value as
>>>>>>> Add descriptions and test protected objects
        well as clear the internal atm attribute.
        """
        scenario = Scenario('test', 0, 0.0, original_ground_range)
        self.check_scenario(scenario, 'test', 0, 0.0, original_ground_range)
        scenario._atm = np.array([0])
        scenario.ground_range = new_ground_range
        assert scenario._atm is None
        assert scenario._ground_range != original_ground_range
        assert scenario.ground_range != original_ground_range
        self.check_scenario(scenario, 'test', 0, 0.0, new_ground_range)

    @pytest.mark.parametrize('name, ihaze, altitude, ground_range', [
        ('test', -1, 1000.0, 0.0),
        ('test', 0, 1000.0, 0.0),
        ('test', 1, 1.0, 1.0),
    ])
    def test_atm_index_error(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float
    ) -> None:
        """
        Cover cases where IndexError occurs
        """
        with pytest.raises(IndexError):
            scenario = Scenario(name, ihaze, altitude, ground_range)
            scenario.atm

    @pytest.mark.parametrize('name, ihaze, altitude, ground_range, expected', [
        ('test', 2, 1000.0, 0.0, utils.loadDatabaseAtmosphere(1000.0, 0.0, 2)),
        ('test', 1, 1000.0, 0.0, utils.loadDatabaseAtmosphere(1000.0, 0.0, 1)),
        ('test', 1, 1000.0, 5.0, utils.loadDatabaseAtmosphere(1000.0, 5.0, 1)),
        ('test', 1, 2000.0, 0.0, utils.loadDatabaseAtmosphere(2000.0, 0.0, 1)),
    ])
    def test_atm(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        expected: np.ndarray
    ) -> None:
        """
        Test atm with expected inputs and outputs as well as checking _atm attribute
        is set properly.
        """
        scenario = Scenario(name, ihaze, altitude, ground_range)
        assert scenario._atm is None
        atm = scenario.atm
        assert scenario._atm is not None
        assert np.isclose(atm, expected).all()
