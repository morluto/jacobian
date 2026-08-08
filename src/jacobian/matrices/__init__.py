"""Artifact-backed HNF, linear-system, and FLINT-worker operations.

This package owns the non-pilot matrix operations that require durable
artifact identity: Python-FLINT Hermite normal forms, exact rational
linear-system solution/inconsistency producers, and their independent
checkers.  Inline matrix operations (determinant, rank, RREF, nullspace,
product, inverse, trace, characteristic polynomial, Smith normal form,
adjugate, rational linear solve, and LLL lattice reduction) live in the
domain-bundle pilot under ``jacobian.domains.matrix_lattice`` and are
not duplicated here.
"""

from jacobian.matrices.linear_capabilities import (
    LinearRationalInconsistencyCheckerInstallation,
    LinearRationalSolutionCheckerInstallation,
    install_linear_rational_inconsistency_checker,
    install_linear_rational_solution_checker,
)
from jacobian.matrices.normal_form import (
    MatrixNormalFormCheckerInstallation,
    install_matrix_normal_form_checker,
)

__all__ = [
    "LinearRationalInconsistencyCheckerInstallation",
    "LinearRationalSolutionCheckerInstallation",
    "MatrixNormalFormCheckerInstallation",
    "install_linear_rational_inconsistency_checker",
    "install_linear_rational_solution_checker",
    "install_matrix_normal_form_checker",
]
