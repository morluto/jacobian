"""Typed wire contracts for sum-of-squares operations."""

from __future__ import annotations

from math import ceil, log10
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"sum_of_squares.{reason}", message)


def _require_bounded_polynomial(
    poly: RationalPolynomial, label: str, *, max_terms: int
) -> None:
    """Bound one input polynomial's terms, total degree, and coefficients."""
    terms = poly.polynomial.terms
    if len(terms) > max_terms:
        raise _validation_error(
            "term_bound", f"{label} exceeds the {max_terms}-term bound"
        )
    for term in terms:
        if sum(term.exponents) > MAX_SOS_DEGREE:
            raise _validation_error(
                "degree_bound", f"{label} exceeds total-degree bound"
            )
        coeff = term.coefficient
        if max(len(coeff.num.lstrip("-")), len(coeff.den)) > MAX_SOS_COEFF_DIGITS:
            raise _validation_error(
                "coefficient_bound", f"{label} coefficient exceeds digit bound"
            )


def _require_monomial_basis(basis: tuple[RationalPolynomial, ...]) -> None:
    """Require distinct unit-coefficient single-term monomials."""
    seen: set[tuple[int, ...]] = set()
    for idx, entry in enumerate(basis):
        terms = entry.polynomial.terms
        if len(terms) != 1:
            raise _validation_error(
                "basis_monomial", f"basis[{idx}] must be a single-term monomial"
            )
        if terms[0].coefficient.as_fraction() != 1:
            raise _validation_error(
                "basis_coefficient", f"basis[{idx}] must have unit coefficient"
            )
        if terms[0].exponents in seen:
            raise _validation_error(
                "basis_distinct", "monomial basis entries must be distinct"
            )
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
            raise _validation_error(
                "ring_mismatch", "all summands must use the same ring as the polynomial"
            )
    predicted = sum(len(s.polynomial.terms) ** 2 for s in summands)
    if predicted > MAX_SOS_PREDICTED_TERMS:
        raise _validation_error(
            "sos_work_bound", "predicted SOS expansion exceeds term bound"
        )
    # A coefficient in the expansion combines at most ``predicted`` products.
    # Each product has a numerator and denominator no wider than twice its
    # input coefficient width.  Bounding the unreduced common denominator is
    # deliberately conservative, but means parsing never needs to expand a
    # polynomial merely to discover that its canonical result cannot fit.
    max_digits = max(
        (
            max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
            for summand in summands
            for term in summand.polynomial.terms
        ),
        default=1,
    )
    coefficient_digits = 2 * max_digits * predicted + ceil(log10(predicted + 1))
    if coefficient_digits > 32_768:
        raise _validation_error(
            "coefficient_growth_bound",
            "predicted SOS coefficient growth exceeds the canonical rational limit",
        )


def _require_square_gram_side(
    entries: tuple[tuple[CanonicalRational, ...], ...], side: int
) -> None:
    if len(entries) != side or any(len(row) != side for row in entries):
        raise _validation_error(
            "gram_shape",
            "gram_matrix must be square with side equal to monomial_basis length",
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
        raise _validation_error("gram_dimension", "gram matrix dimension exceeds bound")
    # Every matrix coefficient is bounded like every polynomial coefficient;
    # otherwise a single 32,768-digit entry could drive unbounded
    # characteristic-polynomial arithmetic inside the exact PSD check.
    try:
        require_matrix_scalar_digits(
            gram_matrix.entries,
            maximum=MAX_SOS_COEFF_DIGITS,
            label="gram_matrix",
        )
    except ValueError as exc:
        raise _validation_error("coefficient_bound", str(exc)) from exc
    # Predicted reconstruction terms for z^T Q z
    max_basis_terms = max(len(b.polynomial.terms) for b in monomial_basis)
    predicted = len(gram_matrix.entries) ** 2 * max(1, max_basis_terms**2)
    if predicted > MAX_SOS_PREDICTED_TERMS * 4:
        raise _validation_error(
            "gram_work_bound", "predicted Gram reconstruction exceeds term bound"
        )


class SOSDecompositionCheckRequest(StrictModel):
    """Check that p = q_1^2 + ... + q_r^2 over QQ.

    The summand family may be empty: its exact sum is the canonical zero
    polynomial.  This makes the zero decomposition available without an
    artificial zero witness.
    """

    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(max_length=64)


class SOSDecompositionCheckResult(StrictModel):
    """Whether the decomposition is exact."""

    is_valid: bool
    polynomial: RationalPolynomial
    summands: tuple[RationalPolynomial, ...] = Field(max_length=64)
    computed_sum: RationalPolynomial

    @model_validator(mode="after")
    def bind_sos(self) -> Self:
        if self.computed_sum.variables != self.polynomial.variables:
            raise _validation_error(
                "ring_mismatch", "computed_sum must use the polynomial ring"
            )
        if self.is_valid != (self.computed_sum == self.polynomial):
            raise _validation_error(
                "sos_validity",
                "is_valid must agree with whether computed_sum equals polynomial",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: RationalPolynomial,
        summands: tuple[RationalPolynomial, ...],
        is_valid: bool,
        computed_sum: RationalPolynomial,
    ) -> Self:
        """Construct a result emitted by the owner-local exact kernel."""

        return cls.model_construct(
            is_valid=is_valid,
            polynomial=polynomial,
            summands=summands,
            computed_sum=computed_sum,
        )


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
            raise _validation_error(
                "ring_mismatch", "monomial basis must use the polynomial ring"
            )
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

    @model_validator(mode="after")
    def bind_invariants(self) -> Self:
        if self.is_valid != (
            self.is_symmetric and self.reconstructs_polynomial and self.is_psd
        ):
            raise _validation_error(
                "gram_validity", "is_valid must be the conjunction of the three checks"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial: RationalPolynomial,
        monomial_basis: tuple[RationalPolynomial, ...],
        gram_matrix: RationalMatrix,
        is_symmetric: bool,
        reconstructs_polynomial: bool,
        is_psd: bool,
    ) -> Self:
        """Construct a result emitted by the owner-local exact kernel."""

        return cls.model_construct(
            is_valid=is_symmetric and reconstructs_polynomial and is_psd,
            is_symmetric=is_symmetric,
            reconstructs_polynomial=reconstructs_polynomial,
            is_psd=is_psd,
            polynomial=polynomial,
            monomial_basis=monomial_basis,
            gram_matrix=gram_matrix,
        )


__all__ = [
    "GramCertificateRequest",
    "GramCertificateResult",
    "SOSDecompositionCheckRequest",
    "SOSDecompositionCheckResult",
]
