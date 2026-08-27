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
from jacobian.math.geometry.boxes.values import RationalAxisAlignedBox


def _union_volume_from_source(
    source: BoxUnionVolumeRequest,
) -> BoxUnionVolumeResult:
    """Compute the complete ledger for one already-admitted box family."""

    records, union_volume = complete_intersection_ledger(source.boxes)
    return BoxUnionVolumeResult._from_kernel(
        source=source,
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


def compute_box_union_volume(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> BoxUnionVolumeResult:
    """Return exact union volume and the complete inclusion-exclusion ledger.

    Accepts the canonical ordered ``RationalAxisAlignedBox`` family and admits
    it against the published box-union execution envelope before computing.
    """

    return _union_volume_from_source(BoxUnionVolumeRequest(boxes=boxes))


def _box_union_volume_from_request(
    request: BoxUnionVolumeRequest,
) -> BoxUnionVolumeResult:
    """Run one parsed catalog request without reconstructing admission."""

    return _union_volume_from_source(request)


__all__ = ["compute_box_union_volume"]
