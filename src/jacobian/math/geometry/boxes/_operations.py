"""Exact operations on finite families of rational boxes."""

from __future__ import annotations

from jacobian.math.geometry.boxes._kernel import (
    complete_intersection_ledger,
    wire_rational,
)
from jacobian.math.geometry.boxes._models import (
    BoxIntersectionLedgerEntry,
    BoxUnionVolumeRequest,
    BoxUnionVolumeResult,
)


def compute_box_union_volume(
    request: BoxUnionVolumeRequest,
) -> BoxUnionVolumeResult:
    """Return exact union volume and the complete inclusion-exclusion ledger."""

    records, union_volume = complete_intersection_ledger(request.boxes)
    return BoxUnionVolumeResult(
        source=request,
        intersections=tuple(
            BoxIntersectionLedgerEntry(
                box_indices=record.box_indices,
                intersection=record.intersection,
                volume=wire_rational(record.volume),
            )
            for record in records
        ),
        union_volume=wire_rational(union_volume),
    )


__all__ = ["compute_box_union_volume"]
