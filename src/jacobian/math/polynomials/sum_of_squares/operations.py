"""Domain-owned sum-of-squares operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import sympy
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.sum_of_squares._models import (
    GramCertificateResult,
    SOSDecompositionCheckResult,
    _require_bounded_gram_admission,
    _require_bounded_sos_work,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

if TYPE_CHECKING:
    from flint import fmpq_mpoly


def _sparse_polynomial(polynomial: RationalPolynomial) -> fmpq_mpoly:
    from flint import fmpq, fmpq_mpoly_ctx

    context = fmpq_mpoly_ctx.get(polynomial.variables, "lex")
    return context.from_dict(
        {
            term.exponents: fmpq(term.coefficient.num, term.coefficient.den)
            for term in polynomial.polynomial.terms
        }
    )


def _admit(operation: Callable[[], None]) -> None:
    try:
        operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc


def _compute_sos_sum(
    polynomial: RationalPolynomial,
    summands: tuple[RationalPolynomial, ...],
) -> tuple[bool, RationalPolynomial]:
    """Compute the exact sum and coefficient-identity outcome."""
    from flint import fmpq_mpoly_ctx

    context = fmpq_mpoly_ctx.get(polynomial.variables, "lex")
    total = context.constant(0)
    for summand in summands:
        request_checkpoint("during sparse sum-of-squares reconstruction")
        value = _sparse_polynomial(summand)
        total += value * value
    is_valid = _sparse_polynomial(polynomial) == total
    if is_valid:
        computed_sum = polynomial
    else:
        computed_sum = RationalPolynomial(
            variables=polynomial.variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_integer_ratio(
                            int(coefficient.p), int(coefficient.q)
                        ),
                        exponents=tuple(int(exponent) for exponent in exponents),
                    )
                    for exponents, coefficient in total.terms()
                )
            ),
        )
    return is_valid, computed_sum


def check_sos_decomposition(
    polynomial: RationalPolynomial,
    summands: tuple[RationalPolynomial, ...],
) -> SOSDecompositionCheckResult:
    """Check that p = q_1^2 + ... + q_r^2 by exact coefficient identity."""
    _admit(lambda: _require_bounded_sos_work(polynomial, summands))
    is_valid, computed_sum = _compute_sos_sum(polynomial, summands)
    return SOSDecompositionCheckResult._from_kernel(
        is_valid=is_valid,
        polynomial=polynomial,
        summands=summands,
        computed_sum=computed_sum,
    )


def _compute_gram_checks(
    polynomial: RationalPolynomial,
    monomial_basis: tuple[RationalPolynomial, ...],
    gram_matrix: tuple[tuple[CanonicalRational, ...], ...],
) -> tuple[bool, bool, bool]:
    """Compute the exact Gram-certificate checks."""
    matrix = sympy.Matrix(
        [[sympy.Rational(c.as_fraction()) for c in row] for row in gram_matrix]
    )
    is_symmetric = matrix == matrix.T
    from flint import fmpq, fmpq_mpoly_ctx

    context = fmpq_mpoly_ctx.get(polynomial.variables, "lex")
    basis = tuple(_sparse_polynomial(m) for m in monomial_basis)
    reconstructed = context.constant(0)
    for i, row in enumerate(gram_matrix):
        request_checkpoint("during sparse Gram reconstruction")
        for j, coefficient in enumerate(row):
            if coefficient.num:
                reconstructed += (
                    fmpq(coefficient.num, coefficient.den) * basis[i] * basis[j]
                )
    reconstructs = reconstructed == _sparse_polynomial(polynomial)
    eigen_matrix = matrix if is_symmetric else (matrix + matrix.T) / 2
    is_psd = _exact_psd(eigen_matrix)
    return is_symmetric, reconstructs, is_psd


def _exact_psd(matrix: sympy.Matrix) -> bool:
    """Total exact PSD test for a symmetric rational matrix.

    Symmetric Gaussian elimination (Lagrange): a symmetric matrix over QQ
    is positive semidefinite iff every pivot is nonnegative and a zero
    pivot forces the whole remaining row and column to vanish. Unlike an
    eigenvalue computation this terminates on every input — irreducible
    characteristic polynomials raise no backend exception.
    """
    n = matrix.rows
    work = [[sympy.Rational(matrix[i, j]) for j in range(n)] for i in range(n)]
    for k in range(n):
        pivot = work[k][k]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(work[k][j] != 0 for j in range(k + 1, n)) or any(
                work[j][k] != 0 for j in range(k + 1, n)
            ):
                return False
            continue
        for i in range(k + 1, n):
            factor = work[i][k] / pivot
            for j in range(k + 1, n):
                work[i][j] -= factor * work[k][j]
    return True


def check_gram_certificate(
    polynomial: RationalPolynomial,
    monomial_basis: tuple[RationalPolynomial, ...],
    gram_matrix: RationalMatrix,
) -> GramCertificateResult:
    """Check that p = z^T Q z with Q symmetric PSD over QQ."""
    _admit(
        lambda: _require_bounded_gram_admission(polynomial, monomial_basis, gram_matrix)
    )
    is_symmetric, reconstructs, is_psd = _compute_gram_checks(
        polynomial, monomial_basis, gram_matrix.entries
    )
    return GramCertificateResult._from_kernel(
        is_symmetric=is_symmetric,
        reconstructs_polynomial=reconstructs,
        is_psd=is_psd,
        polynomial=polynomial,
        monomial_basis=monomial_basis,
        gram_matrix=gram_matrix,
    )


__all__ = [
    "check_gram_certificate",
    "check_sos_decomposition",
]
