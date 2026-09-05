"""Edge-deletion diameter profile operations."""

from jacobian.math.graphs.edge_deletion_diameter_profile._models import (
    EdgeDeletionDiameterEntry,
    EdgeDeletionDiameterProfileResult,
)
from jacobian.math.graphs.edge_deletion_diameter_profile.operations import (
    edge_deletion_diameter_profile,
)

__all__ = [
    "EdgeDeletionDiameterEntry",
    "EdgeDeletionDiameterProfileResult",
    "edge_deletion_diameter_profile",
]
