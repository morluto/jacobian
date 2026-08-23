"""Typed wire contracts for sum-of-squares operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    RationalMatrix,
    require_matrix_scalar_digits,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_SOS_TERMS = 256
MAX_SOS_SUMMAND_TERMS = 64
MAX_SOS_DEGREE = 12
MAX_SOS_COEFF_DIGITS = 128
MAX_SOS_PREDICTED_TERMS = 4096
# The monomial basis fixes the Gram side, so the basis length shares the
# matrices domain's parse-time dimension bound.
MAX_GRAM_DIMENSION = MAX_MATRIX_DIMENSION


def _require_bounded_polynomial(
    poly: RationalPolynomial, label: str, *, max_terms: int
) -> None:
    """Bound one input polynomial's terms, total degree, and coefficients."""
    terms = poly.polynomial.terms
    if len(terms) > max_terms:
        raise ValueError(f"{label} exceeds the {max_terms}-term bound")
    for term in terms:
        if sum(term.exponents) > MAX_SOS_DEGREE:
            raise ValueError(f"{label} exceeds total-degree bound")
        coeff = term.coefficient
        if max(len(coeff.num.lstrip("-")), len(coeff.den)) > MAX_SOS_COEFF_DIGITS:
            raise ValueError(f"{label} coefficient exceeds digit bound")


MAX_GRAM_DIMENSION = 32


def _require_monomial_basis(basis: tuple[RationalPolynomial, ...]) -> None:
    """Require distinct unit-coefficient single-term monomials."""
    seen: set[tuple[int, ...]] = set()
    for idx, entry in enumerate(basis):
        terms = entry.polynomial.terms
        if len(terms) != 1:
            raise ValueError(f"basis[{idx}] must be a single-term monomial")
        if terms[0].coefficient.as_fraction() != 1:
            raise ValueError(f"basis[{idx}] must have unit coefficient")
        if terms[0].exponents in seen:
            raise ValueError("monomial basis entries must be distinct")
        seen.add(terms[0].exponents)


def _require_bounded_sos_work(
    polynomial: RationalPolynomial,
    summands: tuple[RationalPolynomial, ...],
) -> None:
    """Apply the request admission contract to a retained SOS source."""
    # The target polynomial is consumed linearly, so it takes the wider
    # target budget; only the squared summands take the narrower budget.
    _require_bounded_polynomial(polynomial, "polynomial", max_terms=MAX_SOS_TERMS)
    for idx, summand in enumerate(summands):
        _require_bounded_polynomial(
            summand, f"summand[{idx}]", max_terms=MAX_SOS_SUMMAND_TERMS
        )
        if summand.variables != polynomial.variables:
            raise ValueError("all summands must use the same ring as the polynomial")
    predicted = sum(len(s.polynomial.terms) ** 2 for s in summands)
    if predicted > MAX_SOS_PREDICTED_TERMS:
        raise ValueError("predicted SOS expansion exceeds term bound")


def _require_square_gram_side(
    entries: tuple[tuple[CanonicalRational, ...], ...], side: int
) -> None:
    if len(entries) != side or any(len(row) != side for row in entries):
        raise ValueError(
            "gram_matrix must be square with side equal to monomial_basis length"
        )


def _require_bounded_gram_work(
    dimension: int,
    gram_matrix: RationalMatrix,
    monomial_basis: tuple[RationalPolynomial, ...],
) -> None:
    """Bound the exact reconstruction and PSD work for z^T Q z.

    The dimension gate runs first: an oversized payload is rejected before
    any matrix coefficient is inspected.
    """
    if dimension > MAX_GRAM_DIMENSION:
        raise ValueError("gram matrix dimension exceeds bound")
    # Every matrix coefficient is bounded like every polynomial coefficient;
    # otherwise a single 32,768-digit entry could drive unbounded
    # characteristic-polynomial arithmetic inside the exact PSD check.
    require_matrix_scalar_digits(
        gram_matrix.entries,
        maximum=MAX_SOS_COEFF_DIGITS,
        label="gram_matrix",
    )
    # Predicted reconstruction terms for z^T Q z
    max_basis_terms = max(len(b.polynomial.terms) for b in monomial_basis)
    predicted = len(gram_matrix.entries) ** 2 * max(1, max_basis_terms**2)
    if predicted > MAX_SOS_PREDICTED_TERMS * 4:
        raise ValueError("predicted Gram reconstruction exceeds term bound")


class SOSDecompositionCheckRequest(StrictModel):
    """Check that p = q_1^2 + ... + q_r^2 over QQ."""

    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        for summand in self.summands:
            if summand.variables != self.polynomial.variables:
                raise ValueError(
                    "all summands must use the same ring as the polynomial"
                )
        # Bound polynomial expansion before squaring.
        _require_bounded_sos_work(self.polynomial, self.summands)
        # The term-product cap does not bound exact coefficient growth: 64
        # eight-term summands with distinct 128-digit prime denominators can
        # align onto one output coefficient and reduce to a ~65,000-digit
        # denominator. Admission therefore replays the exact expansion so
        # any over-canonical computed_sum fails parsing, not execution.
        from jacobian.math.sum_of_squares._operations import _check_sos_invariants

        try:
            _check_sos_invariants(self.polynomial, self.summands)
        except Exception as exc:
            raise ValueError(
                "SOS expansion leaves the canonical polynomial domain; "
                "supply smaller or better-scaled summand coefficients"
            ) from exc
        return self


class SOSDecompositionCheckResult(StrictModel):
    """Whether the decomposition is exact."""

    is_valid: bool
    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=64)
    computed_sum: RationalPolynomial
    method: Literal["EXACT_COEFFICIENT_IDENTITY"] = "EXACT_COEFFICIENT_IDENTITY"

    @model_validator(mode="after")
    def bind_sos(self) -> Self:
        from jacobian.math.sum_of_squares._operations import _check_sos_invariants

        _require_bounded_sos_work(self.polynomial, self.summands)
        is_valid, computed = _check_sos_invariants(self.polynomial, self.summands)
        if self.is_valid != is_valid:
            raise ValueError("is_valid must match the exact coefficient identity")
        if self.computed_sum != computed:
            raise ValueError("computed_sum must be the exact sum of squares")
        return self


def _require_bounded_gram_admission(
    polynomial: RationalPolynomial,
    monomial_basis: tuple[RationalPolynomial, ...],
    gram_matrix: RationalMatrix,
) -> None:
    """Apply the full bounded Gram-certificate contract to any payload."""
    n = len(monomial_basis)
    # Structural shape gates read only lengths and run before any matrix
    # coefficient is traversed.
    _require_square_gram_side(gram_matrix.entries, n)
    for summand in monomial_basis:
        if summand.variables != polynomial.variables:
            raise ValueError("monomial basis must use the polynomial ring")
    # The public contract is a monomial-basis Gram certificate: z must be
    # a vector of distinct unit-coefficient single-term monomials.
    _require_monomial_basis(monomial_basis)
    # Bound reconstruction work: each polynomial and basis element must be bounded
    _require_bounded_polynomial(polynomial, "polynomial", max_terms=MAX_SOS_TERMS)
    for idx, basis in enumerate(monomial_basis):
        _require_bounded_polynomial(
            basis, f"basis[{idx}]", max_terms=MAX_SOS_SUMMAND_TERMS
        )
    _require_bounded_gram_work(n, gram_matrix, monomial_basis)


class GramCertificateRequest(StrictModel):
    """Check p = z^T Q z with Q symmetric PSD over QQ."""

    polynomial: RationalPolynomial
    monomial_basis: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_GRAM_DIMENSION
    )
    # The matrices domain's canonical value so a producer's serialized
    # RationalMatrix validates unchanged and the returned matrix enters
    # rank, RREF, and characteristic-polynomial consumers unchanged.
    gram_matrix: RationalMatrix

    @model_validator(mode="after")
    def require_square_matrix(self) -> Self:
        _require_bounded_gram_admission(
            self.polynomial, self.monomial_basis, self.gram_matrix
        )
        return self


class GramCertificateResult(StrictModel):
    """Whether the Gram certificate is valid."""

    is_valid: bool
    is_symmetric: bool
    reconstructs_polynomial: bool
    is_psd: bool
    polynomial: RationalPolynomial
    monomial_basis: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_GRAM_DIMENSION
    )
    gram_matrix: RationalMatrix
    method: Literal["EXACT_RATIONAL_ARITHMETIC"] = "EXACT_RATIONAL_ARITHMETIC"

    @model_validator(mode="after")
    def bind_invariants(self) -> Self:
        from jacobian.math.sum_of_squares._operations import _check_gram_invariants

        # Deserialized results replay through the same bounded admission as
        # the request: no unbounded exact reconstruction or eigenvalue work.
        _require_bounded_gram_admission(
            self.polynomial, self.monomial_basis, self.gram_matrix
        )
        is_sym, recon, psd = _check_gram_invariants(
            self.polynomial, self.monomial_basis, self.gram_matrix.entries
        )
        if self.is_symmetric != is_sym:
            raise ValueError("is_symmetric must match the exact symmetry check")
        if self.reconstructs_polynomial != recon:
            raise ValueError(
                "reconstructs_polynomial must match the exact reconstruction check"
            )
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
