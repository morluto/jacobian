"""Typed wire contracts for sum-of-squares operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

MAX_SOS_TERMS = 256
MAX_SOS_SUMMAND_TERMS = 64
MAX_SOS_DEGREE = 12
MAX_SOS_COEFF_DIGITS = 128
MAX_SOS_PREDICTED_TERMS = 4096


class SOSDecompositionCheckRequest(StrictModel):
    """Check that p = q_1^2 + ... + q_r^2 over QQ."""

    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        for summand in self.summands:
            if summand.variables != self.polynomial.variables:
                raise ValueError("all summands must use the same ring as the polynomial")
        # Bound polynomial expansion before squaring.
        def _check_poly(poly: RationalPolynomial, label: str) -> None:
            terms = poly.polynomial.terms
            if len(terms) > MAX_SOS_SUMMAND_TERMS:
                raise ValueError(f"{label} exceeds term bound")
            for term in terms:
                if sum(term.exponents) > MAX_SOS_DEGREE:
                    raise ValueError(f"{label} exceeds total-degree bound")
                coeff = term.coefficient
                if max(len(coeff.num.lstrip("-")), len(coeff.den)) > MAX_SOS_COEFF_DIGITS:
                    raise ValueError(f"{label} coefficient exceeds digit bound")
        _check_poly(self.polynomial, "polynomial")
        for idx, summand in enumerate(self.summands):
            _check_poly(summand, f"summand[{idx}]")
        # Predicted expansion budget: sum of squares can produce up to terms^2 per summand
        predicted = sum(len(s.polynomial.terms) ** 2 for s in self.summands)
        if predicted > MAX_SOS_PREDICTED_TERMS:
            raise ValueError("predicted SOS expansion exceeds term bound")
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
    gram_matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(min_length=1, max_length=64)

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
        # Validate canonical rational entries are bounded (non-evaluating)
        for row in self.gram_matrix:
            for entry in row:
                # CanonicalRational already validates digit bounds and non-evaluating
                pass
        # Bound total matrix size to keep exact PSD check bounded
        if n > 32:
            raise ValueError("gram matrix dimension exceeds bound")
        return self


class GramCertificateResult(StrictModel):
    """Whether the Gram certificate is valid."""

    is_valid: bool
    is_symmetric: bool
    reconstructs_polynomial: bool
    is_psd: bool
    polynomial: RationalPolynomial
    monomial_basis: tuple[RationalPolynomial, ...]
    gram_matrix: tuple[tuple[CanonicalRational, ...], ...]
    method: Literal["EXACT_RATIONAL_ARITHMETIC"] = "EXACT_RATIONAL_ARITHMETIC"

    @model_validator(mode="after")
    def bind_invariants(self) -> Self:
        from jacobian.math.sum_of_squares._operations import _check_gram_invariants

        is_sym, recon, psd = _check_gram_invariants(
            self.polynomial, self.monomial_basis, self.gram_matrix
        )
        if self.is_symmetric != is_sym:
            raise ValueError("is_symmetric must match the exact symmetry check")
        if self.reconstructs_polynomial != recon:
            raise ValueError("reconstructs_polynomial must match the exact reconstruction check")
        if self.is_psd != psd:
            raise ValueError("is_psd must match the exact PSD check")
        if self.is_valid != (is_sym and recon and psd):
            raise ValueError("is_valid must be the conjunction of the three checks")
        return self


__all__ = [
    "GramCertificateRequest",
    "GramCertificateResult",
    "SOSDecompositionCheckRequest",
    "SOSDecompositionCheckResult",
]
