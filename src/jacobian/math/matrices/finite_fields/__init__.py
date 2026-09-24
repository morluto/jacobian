"""Exact finite-dimensional linear algebra over prime fields."""

from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    column_basis,
    nullspace,
    quotient_basis,
    rank,
    rref,
)
from jacobian.math.matrices.finite_fields.operations import (
    project_quotient_vector,
    quotient_space,
    verify_rank,
    verify_rref,
)
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientRequest,
    PrimeFieldQuotientSpace,
    PrimeFieldQuotientVector,
    PrimeFieldSubspace,
    PrimeFieldVectorProjectionRequest,
)

__all__ = [
    "PrimeFieldMatrix",
    "PrimeFieldQuotientRequest",
    "PrimeFieldQuotientSpace",
    "PrimeFieldQuotientVector",
    "PrimeFieldSubspace",
    "PrimeFieldVectorProjectionRequest",
    "column_basis",
    "nullspace",
    "project_quotient_vector",
    "quotient_basis",
    "quotient_space",
    "rank",
    "rref",
    "verify_rank",
    "verify_rref",
]
