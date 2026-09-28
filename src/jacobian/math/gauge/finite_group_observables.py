"""Exact conjugacy observables for finite-table lattice holonomies."""

from __future__ import annotations

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyResult,
    OrientedGaugePath,
)
from jacobian.math.gauge.finite_group import finite_group_gauge_holonomy

MAX_HOLONOMY_CONJUGACY_WORK = 2_000
MAX_HOLONOMY_CONJUGACY_OUTPUT_UNITS = 1_000_000


class FiniteGroupConjugacyProfileRequest(StrictModel):
    """Profile one based loop by computing its finite-table holonomy."""

    field: FiniteGroupGaugeField
    path: OrientedGaugePath


class FiniteGroupConjugacyProfile(StrictModel):
    """Complete conjugacy class of one exact finite-group loop holonomy."""

    loop: FiniteGroupGaugeHolonomyResult
    conjugate_indices: tuple[StrictInt, ...] = Field(max_length=24)
    class_representative_index: StrictInt
    class_size: StrictInt

    @model_validator(mode="after")
    def require_canonical_summary(self) -> FiniteGroupConjugacyProfile:
        authored = self.loop
        try:
            if (
                type(authored) is not FiniteGroupGaugeHolonomyResult
                or type(getattr(getattr(authored, "field", None), "edge_values", None))
                is not tuple
                or type(
                    getattr(
                        getattr(getattr(authored, "field", None), "lattice", None),
                        "vertices",
                        None,
                    )
                )
                is not tuple
                or type(
                    getattr(
                        getattr(getattr(authored, "field", None), "lattice", None),
                        "edges",
                        None,
                    )
                )
                is not tuple
                or type(
                    getattr(
                        getattr(getattr(authored, "field", None), "group", None),
                        "multiplication",
                        None,
                    )
                )
                is not tuple
                or type(
                    getattr(
                        getattr(getattr(authored, "field", None), "group", None),
                        "inverse",
                        None,
                    )
                )
                is not tuple
                or type(getattr(getattr(authored, "path", None), "steps", None))
                is not tuple
                or type(getattr(authored, "contributions", None)) is not tuple
            ):
                raise ValueError("conjugacy profile must retain a canonical loop")
            loop = FiniteGroupGaugeHolonomyResult.model_validate(authored.model_dump())
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise ValueError("conjugacy profile must retain a canonical loop") from None
        if loop != self.loop:
            raise ValueError("conjugacy profile must retain a canonical loop")
        if loop.start != loop.end or (
            loop.path.basepoint is not None and loop.path.basepoint != loop.start
        ):
            raise ValueError("conjugacy profile must retain a closed based loop")
        indices = self.conjugate_indices
        order = len(loop.field.group.multiplication)
        if (
            type(indices) is not tuple
            or not indices
            or any(
                type(index) is not int or not 0 <= index < order for index in indices
            )
            or tuple(sorted(set(indices))) != indices
            or type(self.class_representative_index) is not int
            or self.class_representative_index != indices[0]
            or type(self.class_size) is not int
            or self.class_size != len(indices)
        ):
            raise ValueError("conjugacy profile summary is not canonical for its group")
        return self


def finite_group_holonomy_conjugacy_profile(
    field: FiniteGroupGaugeField,
    path: OrientedGaugePath,
) -> FiniteGroupConjugacyProfile:
    """Return the exact conjugacy orbit of a complete based-loop holonomy."""
    if not isinstance(field, FiniteGroupGaugeField) or not isinstance(
        path, OrientedGaugePath
    ):
        raise OperationDomainValidationError(
            location=("field", "path"),
            code="lattice_gauge.conjugacy.request_values",
            message="conjugacy requires a finite-group field and oriented path",
        )
    try:
        canonical_field = FiniteGroupGaugeField.model_validate(field.model_dump())
        canonical_path = OrientedGaugePath.model_validate(path.model_dump())
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("field", "path"),
            code="lattice_gauge.conjugacy.request_invalid",
            message="conjugacy requires bounded canonical field and path values",
        ) from exc
    loop = finite_group_gauge_holonomy(canonical_field, canonical_path)
    if loop.start != loop.end or (
        loop.path.basepoint is not None and loop.path.basepoint != loop.start
    ):
        raise OperationDomainValidationError(
            location=("path",),
            code="lattice_gauge.conjugacy.open_path",
            message="conjugacy observables require a closed based loop with matching basepoint",
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
    source_units = len(loop.model_dump_json(warnings=False).encode("utf-8"))
    if source_units + 4096 + order * 4 > MAX_HOLONOMY_CONJUGACY_OUTPUT_UNITS:
        raise OperationResourceAdmissionError(
            location=("field", "path"),
            code="lattice_gauge.conjugacy.output_units",
            message="conjugacy profile exceeds its conservative output-unit envelope",
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
