import numpy as np
import pytest

from pybsm import noise


class TestNoise:
    @pytest.mark.parametrize(
        "peRange, bitdepth",
        [
            (0.0, 0.0),
        ],
    )
    def test_quantization_noise_nan(self, peRange: float, bitdepth: float) -> None:
        """
        Cover cases where nan occurs
        """
        output = noise.quantizationNoise(peRange, bitdepth)
        assert np.isnan(output)

    @pytest.mark.parametrize(
        "peRange, bitdepth",
        [
            (1.0, 0.0),
        ],
    )
    def test_quantization_noise_inf(self, peRange: float, bitdepth: float) -> None:
        """
        Cover cases where inf occurs
        """
        output = noise.quantizationNoise(peRange, bitdepth)
        assert np.isinf(output)

    @pytest.mark.parametrize(
        "peRange, bitdepth, expected",
        [
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.2886751345948129),
        ],
    )
    def test_quantization_noise(self, peRange: float, bitdepth: float, expected: float) -> None:
        """
        Test quantizationNoise with normal inputs and expected outputs
        """
        output = noise.quantizationNoise(peRange, bitdepth)
        assert np.isclose(output, expected)
