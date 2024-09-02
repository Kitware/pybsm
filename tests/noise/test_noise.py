from contextlib import nullcontext as does_not_raise
from typing import ContextManager

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import noise


class TestNoise:
    @pytest.mark.parametrize(
        ("kernel", "expectation"),
        [
            (np.array([[0.25, 0.25], [0.25, 0.25]]),
             does_not_raise()),
            (np.array([[0.5], [0.5]]),
             does_not_raise()),
            (np.array([[1.25, 0.25], [0.25, 0.25]]),
             pytest.raises(ValueError, match="Kernel does not sum to 1")),
            (np.array([[-1.25, 0.25], [0.25, 0.25]]),
             pytest.raises(ValueError, match="Kernel does not sum to 1")),
        ],
    )
    def test_noise_gain_valid(self,
                              snapshot: SnapshotAssertion,
                              kernel: np.ndarray,
                              expectation: ContextManager) -> None:
        """Test noise_gain under reasonable inputs and expected outputs."""
        with expectation:
            assert noise.noise_gain(kernel) == snapshot

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth"),
        [
            (0.0, 0.0),
        ],
    )
    def test_quantization_noise_nan(self, pe_range: float, bit_depth: float) -> None:
        """Cover cases where nan occurs."""
        output = noise.quantization_noise(pe_range, bit_depth)
        assert np.isnan(output)

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth"),
        [
            (1.0, 0.0),
        ],
    )
    def test_quantization_noise_inf(self, pe_range: float, bit_depth: float) -> None:
        """Cover cases where inf occurs."""
        output = noise.quantization_noise(pe_range, bit_depth)
        assert np.isinf(output)

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth", "expected"),
        [
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.2886751345948129),
        ],
    )
    def test_quantization_noise(
        self, pe_range: float, bit_depth: float, expected: float
    ) -> None:
        """Test quantization_noise with normal inputs and expected outputs."""
        output = noise.quantization_noise(pe_range, bit_depth)
        assert np.isclose(output, expected)
