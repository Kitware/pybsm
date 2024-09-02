from typing import Dict, Optional

import numpy as np
import pytest

from pybsm.simulation import Sensor


class TestSensor:
    def check_sensor(
        self,
        sensor: Sensor,
        name: str,
        D: float,  # noqa: N803
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
        eta: float = 0.0,
        p_y: Optional[float] = None,
        w_x: Optional[float] = None,
        w_y: Optional[float] = None,
        int_time: float = 1,
        dark_current: float = 0,
        other_irradiance: float = 0.0,
        read_noise: float = 0,
        max_n: int = int(100.0e6),
        max_well_fill: float = 1.0,
        bit_depth: float = 100.0,
        n_tdi: float = 1.0,
        cold_shield_temperature: float = 70.0,
        optics_temperature: float = 270.0,
        optics_emissivity: float = 0.0,
        cold_filter_transmission: float = 1.0,
        cold_filter_temperature: float = 70.0,
        cold_filter_emissivity: float = 0.0,
        s_x: float = 0.0,
        s_y: float = 0.0,
        da_x: float = 0.0,
        da_y: float = 0.0,
        pv: float = 0.0,
        pv_wavelength: float = 0.633e-6,
    ) -> None:
        """Check if created sensor matches expected parameters."""
        assert sensor.name == name
        assert sensor.D == D
        assert sensor.f == f
        assert sensor.p_x == p_x
        if p_y:
            assert sensor.p_y == p_y
        else:
            assert sensor.p_y == p_x
        assert np.equal(sensor.opt_trans_wavelengths.all(), opt_trans_wavelengths.all())
        assert sensor.eta == eta
        if w_x:
            assert sensor.w_x == w_x
        else:
            assert sensor.w_x == p_x
        if w_y:
            assert sensor.w_y == w_y
        else:
            assert sensor.w_y == p_x / sensor.w_x * sensor.p_y
        assert sensor.int_time == int_time
        assert sensor.dark_current == dark_current
        assert sensor.other_irradiance == other_irradiance
        assert sensor.read_noise == read_noise
        assert sensor.max_n == max_n
        assert sensor.max_well_fill == max_well_fill
        assert sensor.bit_depth == bit_depth
        assert sensor.n_tdi == n_tdi
        assert sensor.cold_shield_temperature == cold_shield_temperature
        assert sensor.optics_temperature == optics_temperature
        assert sensor.optics_emissivity == optics_emissivity
        assert sensor.cold_filter_transmission == cold_filter_transmission
        assert sensor.cold_filter_temperature == cold_filter_temperature
        assert sensor.cold_filter_emissivity == cold_filter_emissivity
        assert sensor.s_x == s_x
        assert sensor.s_y == s_y
        assert sensor.da_x == da_x
        assert sensor.da_y == da_y
        assert sensor.pv == pv
        assert sensor.pv_wavelength == pv_wavelength

    @pytest.mark.parametrize(
        ("name", "D", "f", "p_x", "opt_trans_wavelengths", "eta", "other_args"),
        [
            ("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0]), 0.0, {}),
            ("Test", 1.0, 1.0, 1.0, np.array([0.0, 1.0]), 0.0,
             {"p_y": 1.0, "w_x": 1.0, "w_y": 1.0}),
        ],
    )
    def test_initialization(
        self,
        name: str,
        D: float,  # noqa: N803
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
        eta: float,
        other_args: Dict[str, int]
    ) -> None:
        """Check if created sensor matches expected parameters."""
        sensor = Sensor(name, D, f, p_x, opt_trans_wavelengths, eta, **other_args)
        self.check_sensor(sensor, name, D, f, p_x, opt_trans_wavelengths, eta, **other_args)
