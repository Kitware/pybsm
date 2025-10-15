# conftest.py
import io
import json
import math
from typing import Any

import numpy as np
import pytest
from PIL import Image
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension
from syrupy.extensions.single_file import SingleFileSnapshotExtension

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":

    @pytest.fixture(scope="session", autouse=True)
    def set_numpy_printoptions() -> None:
        """Sets global NumPy print options for the entire test session."""
        # Pre numpy2.0, legacy="1.25" is not a valid option
        np.set_printoptions(legacy="1.25")  # pyright: ignore[reportArgumentType]


class FuzzyFloatSnapshotExtension(JSONSnapshotExtension):
    def __init__(self, *, rtol: float = 1e-4, atol: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rtol = rtol
        self.atol = atol

    def serialize(self, data: np.ndarray | float, **_: Any) -> str:  # noqa: C901
        def _serialize_item(item: np.ndarray | complex | tuple | list) -> dict:  # noqa: C901
            if isinstance(item, np.ndarray):
                return {
                    "__type__": "ndarray",
                    "dtype": str(item.dtype),
                    "shape": tuple(int(x) for x in item.shape),
                    "data": [_serialize_item(x) for x in item.tolist()],
                }
            if isinstance(item, (float, int)):
                return {"__type__": "float", "value": float(item)}
            if isinstance(item, complex):
                return {"__type__": "complex", "real": item.real, "imag": item.imag}
            if isinstance(item, tuple):
                return {"__type__": "tuple", "items": [_serialize_item(x) for x in item]}
            if isinstance(item, list):
                return {"__type__": "list", "items": [_serialize_item(x) for x in item]}
            raise TypeError(f"Unsupported type in snapshot serialization: {type(item)}")

        return json.dumps(_serialize_item(data)) + "\n"

    def deserialize(self, data: str) -> np.ndarray | float | complex | tuple | list:  # noqa: C901
        def _decode_special_float(x: float | str) -> float:
            if x == "NaN":
                return float("nan")
            if x == "Infinity":
                return float("inf")
            if x == "-Infinity":
                return float("-inf")
            return float(x)

        def _deserialize_item(obj: dict) -> np.ndarray | float | complex | tuple | list:  # noqa: C901
            if not isinstance(obj, dict):
                raise ValueError("Invalid serialized data format")

            t = obj.get("__type__")
            if t == "ndarray":
                data = [_deserialize_item(x) for x in obj["data"]]
                return np.array(data, dtype=obj["dtype"]).reshape(obj["shape"])
            if t == "float":
                return _decode_special_float(obj["value"])
            if t == "complex":
                return complex(obj["real"], obj["imag"])
            if t == "tuple":
                return tuple(_deserialize_item(x) for x in obj["items"])
            if t == "list":
                return [_deserialize_item(x) for x in obj["items"]]
            raise ValueError(f"Unknown data type: {t}")

        return _deserialize_item(json.loads(data))

    def matches(self, *, serialized_data: str, snapshot_data: str) -> bool:  # noqa: C901
        try:
            expected = self.deserialize(snapshot_data)
            received = self.deserialize(serialized_data)

            def _compare(  # noqa: C901
                a: np.ndarray | complex | tuple | list,
                b: np.ndarray | complex | tuple | list,
            ) -> bool:
                if isinstance(a, float) and isinstance(b, float):
                    return math.isclose(a, b, rel_tol=self.rtol, abs_tol=self.atol)

                if isinstance(a, complex) and isinstance(b, complex):
                    return math.isclose(a.real, b.real, rel_tol=self.rtol, abs_tol=self.atol) and math.isclose(
                        a.imag,
                        b.imag,
                        rel_tol=self.rtol,
                        abs_tol=self.atol,
                    )

                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    return np.allclose(a, b, rtol=self.rtol, atol=self.atol)

                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    if len(a) != len(b):
                        return False
                    return all(_compare(x, y) for x, y in zip(a, b, strict=False))

                # Fallback for other data types
                return a == b

            return _compare(expected, received)
        except ValueError:
            return False


@pytest.fixture
def fuzzy_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(FuzzyFloatSnapshotExtension)


class PSNRImageSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot extension using PSNR metric for image comparison.

    This extension compares images using Peak Signal-to-Noise Ratio (PSNR)
    instead of element-wise numerical comparison. Higher PSNR values indicate
    more similar images. Images pass if their PSNR exceeds a threshold. The
    default threshold of 48.13 corresponds to the psnr for uint8 images
    where each pixel value is off by 1.

    Args:
        min_psnr: Minimum PSNR value in dB required to pass (default: 48.13)
    """

    _file_extension = "tiff"

    def __init__(self, *, min_psnr: float = 48.13, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_psnr = min_psnr

    def serialize(self, data: np.ndarray, **_: Any) -> bytes:
        im = Image.fromarray(data)
        byte_arr = io.BytesIO()
        im.save(byte_arr, format="tiff")
        return byte_arr.getvalue()

    def deserialize(self, data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            # Force load image data to avoid lazy loading
            image.load()
            return np.array(image)

    def matches(self, *, serialized_data: bytes, snapshot_data: bytes) -> bool:
        expected_array = self.deserialize(snapshot_data)
        received_array = self.deserialize(serialized_data)

        # Ensure images have same shape before computing metric
        if expected_array.shape != received_array.shape:
            return False

        # Compute Mean Squared Error
        mse = np.mean((expected_array.astype(float) - received_array.astype(float)) ** 2)

        # To get MAX value, we assume it is a float normalized between 0-1
        # or a uint8
        max_pixel_value = 1.0 if np.issubdtype(received_array.dtype, np.floating) else 255.0

        # If MSE is zero, the images are identical, so PSNR is infinity
        # otherise, PSNR = 10 * log10(MAX^2 / MSE)
        psnr = float(np.inf) if mse == 0.0 else 10 * np.log10((max_pixel_value**2) / mse)

        # Pass if metric value meets or exceeds minimum threshold
        return psnr >= self.min_psnr


@pytest.fixture
def psnr_tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(PSNRImageSnapshotExtension)
