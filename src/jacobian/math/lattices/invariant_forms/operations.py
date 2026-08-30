"""Native invariant integral-form lattice operation."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices.invariant_forms._kernel import (
    invariant_bilinear_form_lattice_kernel,
)
from jacobian.math.lattices.invariant_forms._models import (
    FormKind,
    InvariantBilinearFormLattice,
    RationalMatrixAction,
)


def compute_invariant_bilinear_form_lattice(
    action: RationalMatrixAction,
    kind: FormKind,
) -> InvariantBilinearFormLattice:
    """Compute all integral forms ``Q`` with ``A^T Q A = Q`` exactly."""

    if kind not in ("BILINEAR", "SYMMETRIC", "ALTERNATING"):
        raise OperationDomainValidationError(
            location=("kind",),
            code="lattice.invariant_form.invalid_kind",
            message="kind must be BILINEAR, SYMMETRIC, or ALTERNATING",
        )
    try:
        return invariant_bilinear_form_lattice_kernel(action, kind)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("action",),
            code=exc.type,
            message=exc.message(),
        ) from exc


__all__ = ["compute_invariant_bilinear_form_lattice"]
