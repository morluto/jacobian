"""Operation-time admission for Hochschild computations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology.cohomology.hochschild._models import (
    MAX_ASSOCIATIVITY_DOT_STEPS,
    MAX_HOCHSCHILD_MATRIX_ENTRIES,
    MAX_HOCHSCHILD_TENSOR_ELEMENTS,
    MAX_STRUCTURE_CONSTANT_ENTRIES,
    AlgebraStructure,
    _validation_error,
)
from jacobian.math.topology.cohomology.hochschild._tools import TOOLS


def require_hochschild_budget(dimension: int, max_degree: int) -> None:
    """Reject degrees whose tensor or boundary-matrix budget is exceeded."""
    if dimension ** (max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
        raise _validation_error(
            "hochschild_complex.tensor_budget",
            "requested max_degree exceeds the supported tensor-element budget "
            f"(dimension^{max_degree + 1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})",
        )
    densest_entries = dimension ** (2 * max_degree + 1)
    if densest_entries > MAX_HOCHSCHILD_MATRIX_ENTRIES:
        raise _validation_error(
            "hochschild_complex.matrix_budget",
            "requested max_degree exceeds the supported boundary-matrix "
            f"entry budget (dimension^(2*max_degree+1) = {densest_entries} "
            f"> {MAX_HOCHSCHILD_MATRIX_ENTRIES})",
        )


def require_algebra_admission(algebra: AlgebraStructure) -> None:
    """Check expensive algebra invariants at operation execution time."""
    structure_entries = algebra.dimension**3 + algebra.dimension
    if structure_entries > MAX_STRUCTURE_CONSTANT_ENTRIES:
        raise _validation_error(
            "hochschild_complex.input_budget",
            "structure constants exceed the supported input-entry budget",
        )
    associativity_steps = 2 * algebra.dimension**5
    if associativity_steps > MAX_ASSOCIATIVITY_DOT_STEPS:
        raise _validation_error(
            "hochschild_complex.associativity_budget",
            "algebra dimension exceeds the associativity-admission work budget",
        )
    algebra._require_associative()
    algebra._require_multiplicative_augmentation()


__all__ = ["require_algebra_admission", "require_hochschild_budget"]


ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "hochschild.chain_complex.compute",
        AdmissionDecision.KEEP,
        "exact Hochschild chain complex from algebra structure constants with trivial bimodule",
    ),
    OperationAdmission(
        "hochschild.homology.compute",
        AdmissionDecision.KEEP,
        "exact Hochschild homology via Gaussian elimination over a prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
