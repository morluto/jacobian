"""Exact conjugacy observables for finite-table lattice holonomies."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyRequest,
    FiniteGroupGaugeHolonomyResult,
    OrientedGaugePath,
)
from jacobian.math.gauge.finite_group import finite_group_gauge_holonomy

MAX_HOLONOMY_CONJUGACY_WORK = 2_000
MAX_HOLONOMY_CONJUGACY_OUTPUT_BYTES = 1_000_000


class FiniteGroupConjugacyProfileRequest(StrictModel):
    """Profile one based loop by computing its finite-table holonomy."""

    field: FiniteGroupGaugeField
    path: OrientedGaugePath


class FiniteGroupConjugacyProfile(StrictModel):
    """Complete conjugacy class of one exact finite-group loop holonomy."""

    loop: FiniteGroupGaugeHolonomyResult
    conjugate_indices: tuple[int, ...] = Field(max_length=24)
    class_representative_index: int
    class_size: int


def finite_group_holonomy_conjugacy_profile(
    request: FiniteGroupConjugacyProfileRequest,
) -> FiniteGroupConjugacyProfile:
    """Return the exact conjugacy orbit of a complete based-loop holonomy."""
    if not isinstance(request, FiniteGroupConjugacyProfileRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="lattice_gauge.conjugacy.request_type",
            message="expected a typed finite-group holonomy conjugacy request",
        )
    loop = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=request.field, path=request.path)
    )
    if loop.start != loop.end:
        raise OperationDomainValidationError(
            location=("path",),
            code="lattice_gauge.conjugacy.open_path",
            message="conjugacy observables require a closed based loop",
        )
    group = loop.field.group
    table, inverse = group.multiplication, group.inverse
    order = len(table)
    work_bound = order * order
    if work_bound > MAX_HOLONOMY_CONJUGACY_WORK:
        raise OperationResourceAdmissionError(
            location=("field", "group"),
            code="lattice_gauge.conjugacy.work_bound",
            message="finite-group conjugacy orbit exceeds its work envelope",
        )
    source_bytes = len(loop.model_dump_json(warnings=False).encode("utf-8"))
    if source_bytes + 4096 + order * 4 > MAX_HOLONOMY_CONJUGACY_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("field", "path"),
            code="lattice_gauge.conjugacy.output_bound",
            message="conjugacy profile exceeds its conservative output envelope",
        )
    if order > 24:
        raise OperationResourceAdmissionError(
            location=("field", "group"),
            code="lattice_gauge.conjugacy.class_bound",
            message="conjugacy class exceeds its output envelope",
        )
    orbit = tuple(
        sorted({table[table[g][loop.holonomy.index]][inverse[g]] for g in range(order)})
    )
    return FiniteGroupConjugacyProfile.model_construct(
        loop=loop,
        conjugate_indices=orbit,
        class_representative_index=orbit[0],
        class_size=len(orbit),
    )


__all__ = [
    "FiniteGroupConjugacyProfile",
    "FiniteGroupConjugacyProfileRequest",
    "finite_group_holonomy_conjugacy_profile",
]
