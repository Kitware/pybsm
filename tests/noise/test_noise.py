from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import noise


class TestNoise:
    @pytest.mark.parametrize(
        ("kernel", "expectation"),
        [
            (np.array([[0.25, 0.25], [0.25, 0.25]]), does_not_raise()),
            (np.array([[0.5], [0.5]]), does_not_raise()),
            (np.array([[1.25, 0.25], [0.25, 0.25]]), pytest.raises(ValueError, match="Kernel does not sum to 1")),
            (np.array([[-1.25, 0.25], [0.25, 0.25]]), pytest.raises(ValueError, match="Kernel does not sum to 1")),
        ],
    )
    def test_noise_gain(
        self,
        fuzzy_snapshot: SnapshotAssertion,
        kernel: np.ndarray,
        expectation: AbstractContextManager,
    ) -> None:
        """Test noise_gain against gold standard results and confirm exceptions are appropriately raised."""
        with expectation:
            output = noise.noise_gain(kernel=kernel)
            fuzzy_snapshot.assert_match(output)

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth"),
        [
            (0.0, 0.0),
        ],
    )
    def test_quantization_noise_nan(self, pe_range: float, bit_depth: float) -> None:
        """Cover cases where nan occurs."""
        output = noise.quantization_noise(pe_range=pe_range, bit_depth=bit_depth)
        assert np.isnan(output)

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth"),
        [
            (1.0, 0.0),
        ],
    )
    def test_quantization_noise_inf(self, pe_range: float, bit_depth: float) -> None:
        """Cover cases where inf occurs."""
        output = noise.quantization_noise(pe_range=pe_range, bit_depth=bit_depth)
        assert np.isinf(output)

    @pytest.mark.parametrize(
        ("pe_range", "bit_depth"),
        [
            (0.0, 1.0),
            (1.0, 1.0),
        ],
    )
    def test_quantization_noise(self, pe_range: float, bit_depth: float, fuzzy_snapshot: SnapshotAssertion) -> None:
        """Test quantization_noise with normal inputs and expected outputs."""
        output = noise.quantization_noise(pe_range=pe_range, bit_depth=bit_depth)
        fuzzy_snapshot.assert_match(output)
