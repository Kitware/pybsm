# conftest.py
import numpy as np
import pytest

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":

    @pytest.fixture(scope="session", autouse=True)
    def set_numpy_printoptions() -> None:
        """Sets global NumPy print options for the entire test session."""
        # Pre numpy2.0, legacy="1.25" is not a valid option
        np.set_printoptions(legacy="1.25")  # pyright: ignore[reportArgumentType]
