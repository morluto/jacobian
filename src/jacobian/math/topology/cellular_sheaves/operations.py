"""Native entry point for cellular sheaf construction from cover maps."""

from __future__ import annotations

from jacobian.math.topology.cellular_sheaves._kernel import (
    from_cover_maps as _from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import (
    FromCoverMapsRequest,
    FromCoverMapsResult,
)


def from_cover_maps(request: FromCoverMapsRequest) -> FromCoverMapsResult:
    """Construct one bounded cellular sheaf or its first typed obstruction.

    Admission and every mathematical check run once inside the kernel; the
    canonical result value carries the complete checked diagram without
    validators replaying commutativity.
    """

    return _from_cover_maps(request)


__all__ = ["from_cover_maps"]
