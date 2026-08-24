"""Exact rational axis-aligned box operations."""

from jacobian.math.geometry.boxes._models import (
    BoxIntersectionLedgerEntry,
    BoxUnionVolumeRequest,
    BoxUnionVolumeResult,
)
from jacobian.math.geometry.boxes._operations import compute_box_union_volume
from jacobian.math.geometry.boxes.values import (
    RationalAxisAlignedBox,
    RationalClosedInterval,
)

__all__ = [
    "BoxIntersectionLedgerEntry",
    "BoxUnionVolumeRequest",
    "BoxUnionVolumeResult",
    "RationalAxisAlignedBox",
    "RationalClosedInterval",
    "compute_box_union_volume",
]
