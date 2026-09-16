"""Native exact integer relation-lattice operation."""

from __future__ import annotations

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups._kernel import (
    compute_relation_lattice_data,
)
from jacobian.math.affine_semigroups._models import (
    MAX_RELATION_LATTICE_DIMENSION,
    MAX_RELATION_LATTICE_INPUT_DIGITS,
    RelationLatticeResult,
)
from jacobian.math.matrices.values import IntegerMatrix


def _admit_relation_lattice(configuration: IntegerMatrix) -> None:
    """Enforce the published envelope on the native admission boundary.

    Catalog requests are bounded by the request model; native callers bypass
    wire validation, so the same axes and scalar-digit limits are enforced
    here before any Smith or Hermite work starts.
    """

    rows = configuration.row_count
    columns = configuration.column_count
    if not (
        1 <= rows <= MAX_RELATION_LATTICE_DIMENSION
        and 1 <= columns <= MAX_RELATION_LATTICE_DIMENSION
    ):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.relation_lattice.budget_exceeded",
            message=(
                "relation-lattice configuration axes are limited to "
                f"{MAX_RELATION_LATTICE_DIMENSION} rows and columns"
            ),
        )
    limit = 10**MAX_RELATION_LATTICE_INPUT_DIGITS
    if any(abs(int(value)) >= limit for row in configuration.entries for value in row):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.relation_lattice.budget_exceeded",
            message=(
                "relation-lattice configuration scalars are limited to "
                f"{MAX_RELATION_LATTICE_INPUT_DIGITS} decimal digits"
            ),
        )


def relation_lattice(configuration: IntegerMatrix) -> RelationLatticeResult:
    """Return the canonical integer kernel lattice ``ker_Z(A)``.

    Admission bounds the configuration axes and scalar digits at both the
    request model and this native boundary; the maintained Smith and Hermite
    kernels bound their own intermediate growth and exact output.
    """

    _admit_relation_lattice(configuration)
    data = compute_relation_lattice_data(configuration)
    return RelationLatticeResult._from_kernel(
        configuration=configuration,
        relation_lattice=data.relation_lattice,
        relation_basis=data.relation_basis,
        hnf_transformation=data.hnf_transformation,
        rank=data.rank,
        nullity=data.nullity,
        smith_invariant_factors=data.smith_invariant_factors,
        smith_rank=data.smith_rank,
        saturated_basis=data.saturated_basis,
        saturation_inclusion_transform=data.saturation_inclusion_transform,
        saturation_index=data.saturation_index,
    )


__all__ = ["relation_lattice"]
