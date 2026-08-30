"""Supported exact matrix API."""

from jacobian.math.matrices.operations import (
    adjugate,
    characteristic_polynomial,
    determinant,
    inverse,
    kronecker_product,
    multiply,
    partial_trace,
    permanent,
    rank,
    rref,
    smith_normal_form,
    solve_linear_system,
    trace,
)
from jacobian.math.matrices.values import (
    SmithNormalForm,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)

__all__ = [
    "SmithNormalForm",
    "SparseRationalMatrix",
    "SparseRationalMatrixEntry",
    "adjugate",
    "characteristic_polynomial",
    "determinant",
    "inverse",
    "kronecker_product",
    "multiply",
    "partial_trace",
    "permanent",
    "rank",
    "rref",
    "smith_normal_form",
    "solve_linear_system",
    "trace",
]
