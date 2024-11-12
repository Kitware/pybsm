import math

from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.amber import AmberSnapshotExtension


class CustomSnapshotExtension(AmberSnapshotExtension):
    def matches(self, *, serialized_data: float, snapshot_data: SnapshotAssertion) -> bool:
        try:
            # Convert both the serialized (new) and snapshot (saved) data to floats
            a = float(serialized_data)
            b = float(snapshot_data)
            # Use math.isclose to compare within a relative tolerance
            return math.isclose(a, b, rel_tol=1e-5)
        except ValueError:
            # Return False if the data cannot be converted to floats
            return False
