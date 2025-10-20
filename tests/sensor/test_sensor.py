from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pybsm.simulation import Sensor


class TestSensor:
    def create_default_sensor(self, other_args: dict[str, Any] | None = None) -> Sensor:
        name = "test"
        d = 1.0
        f = 1.0
        px = 1.0
        opt_trans_wavelengths = np.array([0.0, 1.0])
        if not other_args:
            return Sensor(
                name=name,
                D=d,
                f=f,
                p_x=px,
                opt_trans_wavelengths=opt_trans_wavelengths,
            )
        return Sensor(
            name=name,
            D=d,
            f=f,
            p_x=px,
            opt_trans_wavelengths=opt_trans_wavelengths,
            **other_args,
        )

    def check_sensor(
        self,
        sensor: Sensor,
        name: str,
        d: float,
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
    ) -> None:
        """Check if created sensor matches expected parameters."""
        assert sensor.name == name
        assert d == sensor.D
        assert sensor.f == f
        assert sensor.p_x == p_x
        assert np.equal(sensor.opt_trans_wavelengths.all(), opt_trans_wavelengths.all())

    def test_default_eta(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.eta == 0.0

    def test_provided_eta(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"eta": value})
        assert sensor.eta == value

    def test_default_p_y(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.p_y == sensor.p_x

    def test_provided_p_y(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"p_y": value})
        assert sensor.p_y == value

    def test_default_w_x(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.w_x == sensor.p_x

    def test_provided_w_x(self) -> None:
        value = 10.0
        sensor = self.create_default_sensor({"w_x": value})
        assert sensor.w_x == value

    def test_default_w_y(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.w_y == sensor.p_x / sensor.w_x * sensor.p_y

    def test_provided_w_y(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"w_y": value})
        assert sensor.w_y == value

    def test_default_int_time(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.int_time == 1.0

    def test_provided_int_time(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"int_time": value})
        assert sensor.int_time == value

    def test_default_dark_current(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.dark_current == 0.0

    def test_provided_dark_current(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"dark_current": value})
        assert sensor.dark_current == value

    def test_default_other_irradiance(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.other_irradiance == 0.0

    def test_provided_other_irradiance(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"other_irradiance": value})
        assert sensor.other_irradiance == value

    def test_default_read_noise(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.read_noise == 0.0

    def test_provided_read_noise(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"read_noise": value})
        assert sensor.read_noise == value

    def test_default_max_n(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.max_n == int(100.0e6)

    def test_provided_max_n(self) -> None:
        value = int(100e5)
        sensor = self.create_default_sensor({"max_n": value})
        assert sensor.max_n == value

    def test_default_max_well_fill(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.max_well_fill == 1.0

    def test_provided_max_well_fill(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"max_well_fill": value})
        assert sensor.max_well_fill == value

    def test_default_bit_depth(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.bit_depth == 100.0

    def test_provided_bit_depth(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"bit_depth": value})
        assert sensor.bit_depth == value

    def test_default_n_tdi(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.n_tdi == 1.0

    def test_provided_n_tdi(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"n_tdi": value})
        assert sensor.n_tdi == value

    def test_default_cold_shield_temperature(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.cold_shield_temperature == 70.0

    def test_provided_cold_shield_temperature(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"cold_shield_temperature": value})
        assert sensor.cold_shield_temperature == value

    def test_default_optics_temperature(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.optics_temperature == 270.0

    def test_provided_optics_temperature(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"optics_temperature": value})
        assert sensor.optics_temperature == value

    def test_default_optics_emissivity(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.optics_emissivity == 0.0

    def test_provided_optics_emissivity(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"optics_emissivity": value})
        assert sensor.optics_emissivity == value

    def test_default_cold_filter_transmission(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.cold_filter_transmission == 1.0

    def test_provided_cold_filter_transmission(self) -> None:
        value = 0.0
        sensor = self.create_default_sensor({"cold_filter_transmission": value})
        assert sensor.cold_filter_transmission == value

    def test_default_cold_filter_temperature(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.cold_filter_temperature == 70.0

    def test_provided_cold_filter_temperature(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"cold_filter_temperature": value})
        assert sensor.cold_filter_temperature == value

    def test_default_cold_filter_emissivity(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.cold_filter_emissivity == 0.0

    def test_provided_cold_filter_emissivity(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"cold_filter_emissivity": value})
        assert sensor.cold_filter_emissivity == value

    def test_default_s_x(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.s_x == 0.0

    def test_provided_s_x(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"s_x": value})
        assert sensor.s_x == value

    def test_default_s_y(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.s_y == 0.0

    def test_provided_s_y(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"s_y": value})
        assert sensor.s_y == value

    def test_default_da_x(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.da_x == 0.0

    def test_provided_da_x(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"da_x": value})
        assert sensor.da_x == value

    def test_default_da_y(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.da_y == 0.0

    def test_provided_da_y(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"da_y": value})
        assert sensor.da_y == value

    def test_default_pv(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.pv == 0.0

    def test_provided_pv(self) -> None:
        value = 1.0
        sensor = self.create_default_sensor({"pv": value})
        assert sensor.pv == value

    def test_default_pv_wavelength(self) -> None:
        sensor = self.create_default_sensor()
        assert sensor.pv_wavelength == 0.633e-6

    def test_provided_pv_wavelength(self) -> None:
        value = 0.634e-5
        sensor = self.create_default_sensor({"pv_wavelength": value})
        assert sensor.pv_wavelength == value

    @pytest.mark.parametrize(
        ("name", "d", "f", "p_x", "opt_trans_wavelengths"),
        [
            ("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
            ("Test", 0.5, 0.5, 0.5, np.array([0.0, 1.0])),
        ],
    )
    def test_initialization(
        self,
        name: str,
        d: float,
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
    ) -> None:
        """Check if created sensor matches expected parameters."""
        sensor = Sensor(
            name=name,
            D=d,
            f=f,
            p_x=p_x,
            opt_trans_wavelengths=opt_trans_wavelengths,
        )
        self.check_sensor(sensor, name, d, f, p_x, opt_trans_wavelengths)

    def test_identical_sensors_same_hash(self) -> None:
        """Verify identical sensors produce same hash."""
        sensor1 = Sensor(
            name="test",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
            eta=0.1,
            p_y=0.01,
            w_x=0.009,
            w_y=0.009,
            s_x=1e-6,
            s_y=1e-6,
            da_x=0.0,
            da_y=0.0,
            pv=0.1,
            pv_wavelength=0.5e-6,
        )
        sensor2 = Sensor(
            name="test",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
            eta=0.1,
            p_y=0.01,
            w_x=0.009,
            w_y=0.009,
            s_x=1e-6,
            s_y=1e-6,
            da_x=0.0,
            da_y=0.0,
            pv=0.1,
            pv_wavelength=0.5e-6,
        )
        assert hash(sensor1) == hash(sensor2)

    @pytest.mark.parametrize(
        ("param_name", "value1", "value2"),
        [
            ("D", 1.0, 1.5),
            ("f", 2.0, 3.0),
            ("p_x", 0.01, 0.02),
            ("p_y", 0.01, 0.015),
            ("w_x", 0.009, 0.008),
            ("w_y", 0.009, 0.008),
            ("eta", 0.1, 0.2),
            ("s_x", 1e-6, 2e-6),
            ("s_y", 1e-6, 2e-6),
            ("da_x", 0.0, 1e-7),
            ("da_y", 0.0, 1e-7),
            ("pv", 0.1, 0.2),
            ("pv_wavelength", 0.5e-6, 0.6e-6),
        ],
    )
    def test_different_parameters_different_hash(
        self,
        param_name: str,
        value1: float,
        value2: float,
    ) -> None:
        """Verify different parameters produce different hash."""
        base_params = {
            "name": "test",
            "D": 1.0,
            "f": 2.0,
            "p_x": 0.01,
            "opt_trans_wavelengths": np.array([0.4e-6, 0.7e-6]),
            "eta": 0.1,
            "p_y": 0.01,
            "w_x": 0.009,
            "w_y": 0.009,
            "s_x": 1e-6,
            "s_y": 1e-6,
            "da_x": 0.0,
            "da_y": 0.0,
            "pv": 0.1,
            "pv_wavelength": 0.5e-6,
        }
        params1 = base_params.copy()
        params1[param_name] = value1
        params2 = base_params.copy()
        params2[param_name] = value2

        sensor1 = Sensor(**params1)
        sensor2 = Sensor(**params2)

        assert hash(sensor1) != hash(sensor2)

    def test_different_array_parameters_different_hash(self) -> None:
        """Verify array parameter changes produce different hash."""
        sensor1 = Sensor(
            name="test",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
        )
        sensor2 = Sensor(
            name="test",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.5e-6, 0.8e-6]),
        )
        assert hash(sensor1) != hash(sensor2)

    def test_hash_consistency(self) -> None:
        """Verify hash consistency across multiple calls."""
        sensor = Sensor(
            name="test",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
        )
        hash1 = hash(sensor)
        hash2 = hash(sensor)
        hash3 = hash(sensor)

        assert hash1 == hash2 == hash3

    def test_hash_used_for_caching(self) -> None:
        """Verify hash works as cache key."""
        sensor1 = Sensor(
            name="test1",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
        )
        sensor2 = Sensor(
            name="test2",
            D=1.0,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
        )
        sensor3 = Sensor(
            name="test3",
            D=1.5,
            f=2.0,
            p_x=0.01,
            opt_trans_wavelengths=np.array([0.4e-6, 0.7e-6]),
        )

        cache: dict[int, str] = {}
        cache[hash(sensor1)] = "result1"
        cache[hash(sensor2)] = "result2"
        cache[hash(sensor3)] = "result3"

        assert hash(sensor1) == hash(sensor2)
        assert cache[hash(sensor1)] == "result2"

        assert hash(sensor3) != hash(sensor1)
        assert cache[hash(sensor3)] == "result3"
