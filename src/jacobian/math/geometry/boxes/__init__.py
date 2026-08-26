"""Exact rational axis-aligned box operations."""

from jacobian.math.geometry.boxes._models import (
    BoxIntersectionLedgerEntry,
    BoxUnionVolumeResult,
)
from jacobian.math.geometry.boxes._operations import compute_box_union_volume
from jacobian.math.geometry.boxes.values import (
    RationalAxisAlignedBox,
    RationalClosedInterval,
)

__all__ = [
    "BoxIntersectionLedgerEntry",
    "BoxUnionVolumeResult",
    "RationalAxisAlignedBox",
    "RationalClosedInterval",
    "compute_box_union_volume",
]
