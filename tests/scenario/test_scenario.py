import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm.simulation import Scenario


class TestScenario:
    def check_scenario(
        self,
        scenario: Scenario,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        aircraft_speed: float = 0,
        target_reflectance: float = 0.15,
        target_temperature: float = 295,
        background_reflectance: float = 0.07,
        background_temperature: float = 293,
        ha_wind_speed: float = 21,
        cn2_at_1m: float = 1.7e-14,
    ) -> None:
        """Check if created scenario matches expected parameters."""
        assert scenario.name == name
        assert scenario.ihaze == ihaze
        assert scenario.altitude == altitude
        assert scenario.ground_range == ground_range
        assert scenario.aircraft_speed == aircraft_speed
        assert scenario.target_reflectance == target_reflectance
        assert scenario.target_temperature == target_temperature
        assert scenario.background_reflectance == background_reflectance
        assert scenario.background_temperature == background_temperature
        assert scenario.ha_wind_speed == ha_wind_speed
        assert scenario.cn2_at_1m == cn2_at_1m

    @pytest.mark.parametrize(
        ("name", "ihaze", "altitude", "ground_range", "interp", "other_args"),
        [
            ("", 0, 0.0, 0.0, False, {}),
            ("test", 1, 1.0, 1.0, True, {}),
            (
                "test",
                1,
                1.0,
                1.0,
                True,
                {
                    "aircraft_speed": 1.0,
                    "target_reflectance": 1.0,
                    "target_temperature": 1.0,
                    "background_reflectance": 1.0,
                    "background_temperature": 1.0,
                    "ha_wind_speed": 1.0,
                    "cn2_at_1m": 1.0,
                },
            ),
        ],
    )
    def test_initialization(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        interp: bool,
        other_args: dict[str, float],
    ) -> None:
        """Test initialization with and without default parameters."""
        scenario = Scenario(
            name=name,
            ihaze=ihaze,
            altitude=altitude,
            ground_range=ground_range,
            interp=interp,
            **other_args,
        )
        self.check_scenario(scenario, name, ihaze, altitude, ground_range, **other_args)

    @pytest.mark.parametrize(
        ("original_ihaze", "new_ihaze"),
        [
            (0, 1),
        ],
    )
    def test_ihaze(self, original_ihaze: int, new_ihaze: int) -> None:
        """Test that setting the ihaze attribute works appropriately.

        Test that setting the ihaze attribute appropriately updates the internal value and clears the
        internal atm attribute.
        """
        scenario = Scenario(
            name="test",
            ihaze=original_ihaze,
            altitude=0.0,
            ground_range=0.0,
        )
        self.check_scenario(scenario, "test", original_ihaze, 0.0, 0.0)
        scenario._atm = np.array([0])
        scenario.ihaze = new_ihaze
        assert scenario._atm is None
        assert scenario._ihaze != original_ihaze
        assert scenario.ihaze != original_ihaze
        self.check_scenario(scenario, "test", new_ihaze, 0.0, 0.0)

    @pytest.mark.parametrize(
        ("original_altitude", "new_altitude"),
        [
            (0.0, 1.0),
        ],
    )
    def test_altitude(self, original_altitude: float, new_altitude: float) -> None:
        """Test that setting the altitude attribute works appropriately.

        Test setting altitude attribute appropriately updates the internal value as well as
        clear the internal atm attribute.
        """
        scenario = Scenario(
            name="test",
            ihaze=0,
            altitude=original_altitude,
            ground_range=0.0,
        )
        self.check_scenario(scenario, "test", 0, original_altitude, 0.0)
        scenario._atm = np.array([0])
        scenario.altitude = new_altitude
        assert scenario._atm is None
        assert scenario._altitude != original_altitude
        assert scenario.altitude != original_altitude
        self.check_scenario(scenario, "test", 0, new_altitude, 0.0)

    @pytest.mark.parametrize(
        ("original_ground_range", "new_ground_range"),
        [
            (0.0, 1.0),
        ],
    )
    def test_ground_range(
        self,
        original_ground_range: float,
        new_ground_range: float,
    ) -> None:
        """Test that setting the ground_range attribute works appropriately.

        Test that setting the ground_range attribute appropriately updates the internal value as well as
        clear the internal atm attribute.
        """
        scenario = Scenario(
            name="test",
            ihaze=0,
            altitude=0.0,
            ground_range=original_ground_range,
        )
        self.check_scenario(scenario, "test", 0, 0.0, original_ground_range)
        scenario._atm = np.array([0])
        scenario.ground_range = new_ground_range
        assert scenario._atm is None
        assert scenario._ground_range != original_ground_range
        assert scenario.ground_range != original_ground_range
        self.check_scenario(scenario, "test", 0, 0.0, new_ground_range)

    @pytest.mark.parametrize(
        ("name", "ihaze", "altitude", "ground_range", "interp"),
        [
            ("test", 0, 1.0, 0.0, True),
            ("test", 0, 30000.0, 0.0, True),
            ("test", 1, 2, -1.0, True),
            ("test", 1, 2, 400000.0, True),
            ("test", -1, 1000.0, 0.0, False),
            ("test", 0, 1000.0, 0.0, False),
            ("test", 1, 1.0, 1.0, False),
        ],
    )
    def test_atm_index_error(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        interp: bool,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            Scenario(  # noqa: B018
                name=name,
                ihaze=ihaze,
                altitude=altitude,
                ground_range=ground_range,
                interp=interp,
            ).atm

    @pytest.mark.parametrize(
        ("name", "ihaze", "altitude", "ground_range", "interp"),
        [
            ("test", 2, 1000.0, 0.0, True),
            ("test", 1, 1000.0, 0.0, True),
            ("test", 1, 1000.0, 5.0, True),
            ("test", 1, 2000.0, 0.0, True),
            ("test", 2, 1000.0, 0.0, False),
            ("test", 1, 1000.0, 0.0, False),
            ("test", 1, 2000.0, 0.0, False),
        ],
    )
    def test_atm(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        interp: bool,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Test atm with expected inputs and outputs as well as checking _atm attribute is set properly."""
        scenario = Scenario(
            name=name,
            ihaze=ihaze,
            altitude=altitude,
            ground_range=ground_range,
            interp=interp,
        )
        assert scenario._atm is None
        atm = scenario.atm
        assert scenario._atm is not None
        fuzzy_snapshot.assert_match(atm)
