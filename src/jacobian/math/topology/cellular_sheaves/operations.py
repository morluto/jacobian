"""Native entry point for cellular sheaf construction from cover maps."""

from __future__ import annotations

from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cellular_sheaves._kernel import (
    from_cover_maps as _from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._kernel import (
    sheaf_cohomology as _sheaf_cohomology,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
    FromCoverMapsResult,
    SheafCohomologyResult,
    SheafField,
    SheafStalk,
)


def from_cover_maps(
    complex_: FiniteSimplicialComplex,
    coefficient_field: SheafField,
    prime: int | None,
    stalks: tuple[SheafStalk, ...],
    cover_maps: tuple[CoverRestrictionMatrix, ...],
) -> FromCoverMapsResult:
    """Construct one bounded cellular sheaf or its first typed obstruction.

    Admission and every mathematical check run once inside the kernel; the
    canonical result value carries the complete checked diagram without
    validators replaying commutativity.
    """

    return _from_cover_maps(complex_, coefficient_field, prime, stalks, cover_maps)


def sheaf_cohomology(sheaf: FiniteCellularSheaf) -> SheafCohomologyResult:
    """Compute cellular cohomology of a checked finite cellular sheaf.

    Admission and every mathematical check run once inside the kernel; the
    canonical result value carries the complete cochain complex without
    validators replaying the coboundary mathematics.
    """

    return _sheaf_cohomology(sheaf)


__all__ = ["from_cover_maps", "sheaf_cohomology"]
