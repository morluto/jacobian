"""Typed wire contracts for sum-of-squares operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial


class SOSDecompositionCheckRequest(StrictModel):
    """Check that p = q_1^2 + ... + q_r^2 over QQ."""

    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        for summand in self.summands:
            if summand.variables != self.polynomial.variables:
                raise ValueError("all summands must use the same ring as the polynomial")
        return self


class SOSDecompositionCheckResult(StrictModel):
    """Whether the decomposition is exact."""

    is_valid: bool
    polynomial: RationalPolynomial
    computed_sum: RationalPolynomial
    method: Literal["EXACT_COEFFICIENT_IDENTITY"] = "EXACT_COEFFICIENT_IDENTITY"


class GramCertificateRequest(StrictModel):
    """Check p = z^T Q z with Q symmetric PSD over QQ."""

    polynomial: RationalPolynomial
    monomial_basis: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=64)
    gram_matrix: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_square_matrix(self) -> Self:
        n = len(self.monomial_basis)
        if len(self.gram_matrix) != n:
            raise ValueError("gram_matrix must be square with side equal to monomial_basis length")
        for row in self.gram_matrix:
            if len(row) != n:
                raise ValueError("gram_matrix must be square")
        for summand in self.monomial_basis:
            if summand.variables != self.polynomial.variables:
                raise ValueError("monomial basis must use the polynomial ring")
        return self


class GramCertificateResult(StrictModel):
    """Whether the Gram certificate is valid."""

    is_valid: bool
    is_symmetric: bool
    reconstructs_polynomial: bool
    is_psd: bool
    method: Literal["EXACT_RATIONAL_ARITHMETIC"] = "EXACT_RATIONAL_ARITHMETIC"


__all__ = [
    "GramCertificateRequest",
    "GramCertificateResult",
    "SOSDecompositionCheckRequest",
    "SOSDecompositionCheckResult",
]
