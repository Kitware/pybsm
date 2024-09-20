import unittest.mock as mock
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import ContextManager

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybsm.simulation import RefImage, Scenario, Sensor

BASE_PATH = Path(__file__).parent.parent.parent
IMAGE_PATH = BASE_PATH / 'examples' / 'data' / 'M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff'


class TestRefImage:
    @pytest.mark.parametrize(
        ("pix_values", "refl_values", "expectation"),
        [
            (None, None, does_not_raise()),
            (np.array([0.05, 0.95]), None,
             pytest.raises(ValueError, match="If 'pix_values' is provided, 'refl_values' must be as well.")),
            (None, np.array([0.05, 0.95]), does_not_raise()),
            (np.array([0.05, 0.95]), np.array([0.1, 0.9]), does_not_raise())

        ]
    )
    def test_ref_image_init(self,
                            pix_values: np.ndarray,
                            refl_values: np.ndarray,
                            expectation: ContextManager) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = (3.19 / 160.0)
        with expectation:
            RefImage(img, gsd, pix_values, refl_values)

    @pytest.mark.parametrize(
            ("altitude", "expected_sensor", "expected_scenario"),
            [(1000,
             Sensor("ref_image",
                    0.10833855799731913,
                    1.0031347962382444,
                    20e-6,
                    np.array([3.8e-07, 7.0e-07])),
             Scenario("ref_image",
                      1,
                      1000,
                      ground_range=0,
                      aircraft_speed=0,
                      ha_wind_speed=0,
                      cn2_at_1m=0)),
             (10000,
             Sensor("ref_image",
                    1.0833855799376628,
                    10.031347962382446,
                    20e-6,
                    np.array([3.8e-07, 7.0e-07])),
             Scenario("ref_image",
                      1,
                      10000,
                      ground_range=0,
                      aircraft_speed=0,
                      ha_wind_speed=0,
                      cn2_at_1m=0))]
    )
    def test_estimate_capture_parameters(self,
                                         altitude: float,
                                         expected_sensor: Sensor,
                                         expected_scenario: Scenario) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = (3.19 / 160.0)
        ref_image = RefImage(img, gsd)
        (sensor, scenario) = ref_image.estimate_capture_parameters(altitude)
        assert sensor.name == expected_sensor.name
        assert sensor.D == expected_sensor.D
        assert sensor.f == expected_sensor.f
        assert sensor.p_x == expected_sensor.p_x
        assert np.equal(sensor.opt_trans_wavelengths.all(),
                        expected_sensor.opt_trans_wavelengths.all())

        assert scenario.name == expected_scenario.name
        assert scenario.ihaze == expected_scenario.ihaze
        assert scenario.altitude == expected_scenario.altitude
        assert scenario.ground_range == expected_scenario.ground_range
        assert scenario.aircraft_speed == expected_scenario.aircraft_speed
        assert scenario.ha_wind_speed == expected_scenario.ha_wind_speed
        assert scenario.cn2_at_1m == expected_scenario.cn2_at_1m

    @mock.patch("matplotlib.pyplot.imshow")
    @mock.patch("matplotlib.pyplot.xlabel")
    @mock.patch("matplotlib.pyplot.ylabel")
    @mock.patch("matplotlib.pyplot.tight_layout")
    def test_show_method(self,
                         mock_tight_layout: mock.MagicMock,
                         mock_ylabel: mock.MagicMock,
                         mock_xlabel: mock.MagicMock,
                         mock_imshow: mock.MagicMock) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = (3.19 / 160.0)
        # Create the instance of TestClass
        obj = RefImage(img, gsd)

        # Call the show method
        obj.show()

        # Check if the imshow was called with correct extent calculation
        h, w = img.shape[:2]
        expected_extent = [
            -w / 2 * gsd,
            w / 2 * gsd,
            -h / 2 * gsd,
            h / 2 * gsd,
        ]
        assert mock_imshow.call_count == 1
        call_args, call_kwargs = mock_imshow.call_args

        # Validate the image argument using numpy.testing
        np.testing.assert_array_equal(call_args[0], img)

        # Validate the extent argument
        assert call_kwargs['extent'] == expected_extent

        # Verify xlabel was called with correct parameters
        mock_xlabel.assert_called_once_with("X-Position (m)", fontsize=24)

        # Verify ylabel was called with correct parameters
        mock_ylabel.assert_called_once_with("Y-Position (m)", fontsize=24)

        # Verify tight_layout was called once
        mock_tight_layout.assert_called_once()
