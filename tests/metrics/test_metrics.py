import pytest

from pybsm.metrics import Metrics


class TestMetrics:
    @pytest.mark.parametrize(
        ("name"),
        [("Test")],
    )
    def test_initialization(
        self,
        name: str,
    ) -> None:
        """Check if created metrics matches expected parameters."""
        metrics = Metrics(name)
        assert name == metrics.name
