import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm.simulation import RefImage

BASE_PATH = Path(__file__).parent.parent.parent
IMAGE_PATH = BASE_PATH / "docs" / "examples" / "data" / "M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


class TestRefImage:
    @pytest.mark.parametrize(
        ("pix_values", "refl_values", "expectation"),
        [
            (None, None, does_not_raise()),
            (
                np.array([0.05, 0.95]),
                None,
                pytest.raises(ValueError, match="If 'pix_values' is provided, 'refl_values' must be as well."),
            ),
            (None, np.array([0.05, 0.95]), does_not_raise()),
            (np.array([0.05, 0.95]), np.array([0.1, 0.9]), does_not_raise()),
        ],
    )
    def test_ref_image_init(
        self,
        pix_values: np.ndarray,
        refl_values: np.ndarray,
        expectation: AbstractContextManager,
    ) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = 3.19 / 160.0
        with expectation:
            ref = RefImage(
                img=img,
                gsd=gsd,
                pix_values=pix_values,
                refl_values=refl_values,
            )
            assert np.array_equal(ref.img, img)
            assert ref.gsd == gsd
            if pix_values is not None:
                assert np.array_equal(ref.pix_values, pix_values)
            if refl_values is not None:
                assert np.array_equal(ref.refl_values, refl_values)

    @pytest.mark.parametrize(("altitude"), [(1000), (10000)])
    def test_estimate_capture_parameters(self, altitude: float, snapshot: SnapshotAssertion) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = 3.19 / 160.0
        ref_image = RefImage(img=img, gsd=gsd)
        output = ref_image.estimate_capture_parameters(altitude=altitude)
        assert output == snapshot

    @mock.patch("pybsm.simulation.ref_image.plt")
    def test_show_method(self, mock_plt: mock.MagicMock, snapshot: SnapshotAssertion) -> None:
        img = plt.imread(IMAGE_PATH)
        gsd = 3.19 / 160.0
        # Create the instance of TestClass
        obj = RefImage(img=img, gsd=gsd)

        # Call the show method
        obj.show()

        # Check if the imshow was called with correct extent calculation
        assert mock_plt.imshow.call_count == 1
        call_args, call_kwargs = mock_plt.imshow.call_args

        # Validate the image argument using numpy.testing
        np.testing.assert_array_equal(call_args[0], img)

        # Validate the extent argument
        assert call_kwargs["extent"] == snapshot

        # Verify xlabel was called with correct parameters
        mock_plt.xlabel.assert_called_once_with("X-Position (m)", fontsize=24)

        # Verify ylabel was called with correct parameters
        mock_plt.ylabel.assert_called_once_with("Y-Position (m)", fontsize=24)

        # Verify tight_layout was called once
        mock_plt.tight_layout.assert_called_once()
